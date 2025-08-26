import gc
import json
import os
import pickle
import random
import shutil
import tempfile
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import dask.dataframe as dd
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torch.sparse
from dask.diagnostics import ProgressBar
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix, csr_matrix, vstack
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
from tqdm.auto import tqdm

# tqdm интеграция с pandas
tqdm.pandas()


# -------------------- Загрузка данных --------------------
def load_train_data(max_parts=0, max_rows=1_000):
    """
    Загружаем parquet-файлы orders, tracker, items, categories_tree, test_users.
    Ищем рекурсивно по папкам все .parquet файлы. Ограничиваем общее количество строк.
    """
    paths = {
        "orders": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_orders_data/",
        "tracker": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_tracker_data/",
        "items": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_items_data/",
        "categories": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train_final_categories_tree/",
        "test_users": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_test_for_participants/test_for_participants/",
    }

    columns_map = {
        "orders": ["item_id", "user_id", "created_timestamp", "last_status"],
        "tracker": ["item_id", "user_id", "timestamp", "action_type"],
        "items": ["item_id", "itemname", "fclip_embed", "catalogid"],
        "categories": ["catalogid", "catalogpath", "ids"],
        "test_users": ["user_id"],
    }

    dtype_profiles = {
        "orders": {
            "user_id": "int32",
            "item_id": "int32",
            "created_timestamp": "datetime64[ns]",
            "last_status": "category",
        },
        "tracker": {
            "user_id": "int32",
            "item_id": "int32",
            "timestamp": "datetime64[ns]",
            "action_type": "category",
        },
        "items": {
            "item_id": "int32",
            "catalogid": "int32",
            "itemname": "string",
            "fclip_embed": "object",
        },
        "categories": {
            "catalogid": "int32",
            "catalogpath": "string",
            "ids": "string",
        },
        "test_users": {"user_id": "int32"},
    }

    def find_parquet_files(folder):
        files = glob(os.path.join(folder, "**", "*.parquet"), recursive=True)
        files.sort()
        return files

    def read_sample(
        folder, columns=None, name="", max_parts=max_parts, max_rows=max_rows
    ):
        files = find_parquet_files(folder)
        if not files:
            log_message(f"{name}: parquet файлы не найдены в {folder}")
            return dd.from_pandas(pd.DataFrame(), npartitions=1)

        current_dtypes = dtype_profiles.get(name, {})

        try:
            # Читаем все файлы
            ddf = dd.read_parquet(
                files,
                engine="pyarrow",
                dtype=current_dtypes,
                gather_statistics=False,
                split_row_groups=True,
            )
        except Exception as e:
            log_message(f"{name}: ошибка при чтении parquet ({e}), пропускаем")
            return dd.from_pandas(pd.DataFrame(), npartitions=1)

        if columns is not None:
            available_cols = [c for c in columns if c in ddf.columns]
            if not available_cols:
                log_message(
                    f"{name}: ни одна из колонок {columns} не найдена, пропускаем"
                )
                return dd.from_pandas(pd.DataFrame(), npartitions=1)
            ddf = ddf[available_cols]

        total_parts = ddf.npartitions

        # Ограничиваем количество партиций если нужно
        if max_parts > 0 and max_parts < total_parts:
            ddf = ddf.partitions[:max_parts]
            used_parts = max_parts
        else:
            used_parts = total_parts

        # Если не нужно ограничивать строки - возвращаем как есть
        if max_rows == 0:
            count = ddf.shape[0].compute()
            mem_estimate = ddf.memory_usage(deep=True).sum().compute() / (1024**2)
            log_message(
                f"{name}: {count:,} строк (использовано {used_parts} из {total_parts} партиций), ~{mem_estimate:.1f} MB"
            )
            return ddf

        # Быстрый способ ограничить количество строк - берем первые max_rows
        # Для этого сначала вычисляем общее количество строк
        total_rows = ddf.shape[0].compute()

        if total_rows <= max_rows:
            # Если строк меньше лимита - возвращаем всё
            count = total_rows
            mem_estimate = ddf.memory_usage(deep=True).sum().compute() / (1024**2)
            log_message(
                f"{name}: {count:,} строк (использовано {used_parts} из {total_parts} партиций), ~{mem_estimate:.1f} MB"
            )
            return ddf
        else:
            # Если строк больше - создаем новый ddf с ограничением
            # Для скорости используем head с последующим преобразованием
            limited_ddf = ddf.head(max_rows, compute=False)
            count = max_rows
            mem_estimate = limited_ddf.memory_usage(deep=True).sum().compute() / (
                1024**2
            )
            log_message(
                f"{name}: {count:,} строк (использовано {used_parts} из {total_parts} партиций), ~{mem_estimate:.1f} MB"
            )
            return limited_ddf

    log_message("Загружаем данные...")
    orders_ddf = read_sample(
        paths["orders"], columns=columns_map["orders"], name="orders"
    )
    tracker_ddf = read_sample(
        paths["tracker"], columns=columns_map["tracker"], name="tracker"
    )
    items_ddf = read_sample(paths["items"], columns=columns_map["items"], name="items")
    categories_ddf = read_sample(
        paths["categories"], columns=columns_map["categories"], name="categories"
    )
    test_users_ddf = read_sample(
        paths["test_users"], columns=columns_map["test_users"], name="test_users"
    )
    log_message("Данные загружены")

    return orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_ddf


# -------------------- Фильтрация данных --------------------
def filter_data(orders_ddf, tracker_ddf, items_ddf):
    """
    Фильтруем: оставляем delivered_orders (позитив) и canceled_orders (негатив),
    а также действия page_view, favorite, to_cart.
    """
    log_message("Фильтрация данных...")

    # Заказы: оставляем только delivered и canceled
    orders_ddf = orders_ddf[
        orders_ddf["last_status"].isin(["delivered_orders", "canceled_orders"])
    ].copy()

    # delivered_orders = 1, canceled_orders = 0
    orders_ddf["target"] = orders_ddf["last_status"].apply(
        lambda x: 1 if x == "delivered_orders" else 0, meta=("target", "int8")
    )

    # Действия
    tracker_ddf = tracker_ddf[
        tracker_ddf["action_type"].isin(["page_view", "favorite", "to_cart"])
    ]

    log_message("Фильтрация завершена")
    return orders_ddf, tracker_ddf, items_ddf


# -------------------- Train/Test split по времени --------------------
def train_test_split_by_time(orders_df, test_size=0.2):
    """
    Деление по глобальной дате: train = первые (1 - test_size) по времени,
    test = последние test_size по времени.
    """
    orders_df = orders_df.copy()
    orders_df["created_timestamp"] = pd.to_datetime(orders_df["created_timestamp"])
    orders_df = orders_df.sort_values("created_timestamp")

    cutoff_idx = int(len(orders_df) * (1 - test_size))
    cutoff_ts = orders_df.iloc[cutoff_idx]["created_timestamp"]

    train_df = orders_df[orders_df["created_timestamp"] <= cutoff_ts]
    test_df = orders_df[orders_df["created_timestamp"] > cutoff_ts]

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        cutoff_ts,
    )


# -------------------- Подготовка взаимодействий --------------------
def prepare_interactions(
    train_orders_df,
    tracker_ddf,
    cutoff_ts_per_user,
    batch_size=300_000_000,
    action_weights=None,
    scale_days=5,
    output_dir="/home/root6/python/e_cup/rec_system/data/processed/prepare_interactions_batches",
):
    log_message("Формируем матрицу взаимодействий по батчам...")

    if action_weights is None:
        action_weights = {"page_view": 2, "favorite": 5, "to_cart": 10}

    os.makedirs(output_dir, exist_ok=True)
    batch_files = []
    ref_time = train_orders_df["created_timestamp"].max()

    # ====== Orders ======
    log_message("... для orders")
    n_rows = len(train_orders_df)
    for start in range(0, n_rows, batch_size):
        batch = train_orders_df.iloc[start : start + batch_size].copy()
        days_ago = (ref_time - batch["created_timestamp"]).dt.days.clip(lower=1)
        time_factor = np.log1p(days_ago / scale_days)
        batch = batch.assign(
            timestamp=batch["created_timestamp"],
            weight=5.0 * time_factor,
            action_type="order",
        )[["user_id", "item_id", "weight", "timestamp", "action_type"]]

        path = os.path.join(output_dir, f"orders_batch_{start}.parquet")
        batch.to_parquet(path, index=False, engine="pyarrow")
        batch_files.append(path)
        del batch
        gc.collect()
        log_message(f"Сохранен orders-батч {start}-{min(start+batch_size, n_rows)}")

    # ====== Tracker ======
    log_message("... для tracker")
    tracker_ddf = tracker_ddf[["user_id", "item_id", "timestamp", "action_type"]]

    # Итерируемся по партициям Dask DataFrame
    n_partitions = tracker_ddf.npartitions
    for partition_id in range(n_partitions):
        # Вычисляем одну партицию
        part = tracker_ddf.get_partition(partition_id).compute()
        part["timestamp"] = pd.to_datetime(part["timestamp"])

        # cutoff_ts_per_user здесь один глобальный timestamp
        cutoff_ts = cutoff_ts_per_user
        mask = part["timestamp"] < cutoff_ts
        part = part.loc[mask]

        if part.empty:
            continue

        aw = part["action_type"].map(action_weights).fillna(0)
        days_ago = (ref_time - part["timestamp"]).dt.days.clip(lower=1)
        time_factor = np.log1p(days_ago / scale_days)
        part = part.assign(weight=aw * time_factor)[
            ["user_id", "item_id", "weight", "timestamp", "action_type"]
        ]

        path = os.path.join(output_dir, f"tracker_part_{partition_id}.parquet")
        part.to_parquet(path, index=False, engine="pyarrow")
        batch_files.append(path)
        del part
        gc.collect()
        log_message(f"Сохранен tracker-партиция {partition_id}")

    log_message("Все батчи сохранены на диск.")
    return batch_files


# -------------------- Глобальная популярность --------------------
def compute_global_popularity(orders_df, cutoff_ts_info):
    """
    Считает популярность товаров на основе ТОЛЬКО тренировочных заказов.

    Args:
        orders_df: Все заказы (до split)
        cutoff_ts_info: либо словарь {user_id: cutoff_ts}, либо один глобальный pd.Timestamp
    """
    log_message("Считаем глобальную популярность на основе тренировочных данных...")

    orders_df = orders_df.copy()
    orders_df["created_timestamp"] = pd.to_datetime(orders_df["created_timestamp"])

    if isinstance(cutoff_ts_info, dict):
        # По каждому пользователю свой cutoff
        train_orders = []
        for user_id, cutoff_ts in cutoff_ts_info.items():
            user_orders = orders_df[
                (orders_df["user_id"] == user_id)
                & (orders_df["created_timestamp"] < cutoff_ts)
            ]
            train_orders.append(user_orders)
        train_orders_df = (
            pd.concat(train_orders, ignore_index=True)
            if train_orders
            else pd.DataFrame(columns=orders_df.columns)
        )

    else:
        # Глобальный cutoff (одна дата для всех)
        cutoff_ts = cutoff_ts_info
        train_orders_df = orders_df[orders_df["created_timestamp"] < cutoff_ts]

    # Считаем популярность только на тренировочных данных
    if train_orders_df.empty:
        log_message("Нет тренировочных заказов для расчёта популярности.")
        return pd.Series(dtype=float)

    pop = (
        train_orders_df.groupby("item_id")["item_id"]
        .count()
        .sort_values(ascending=False)
    )
    popularity = pop / pop.max()
    log_message(
        f"Глобальная популярность рассчитана на {len(train_orders_df)} тренировочных заказах"
    )
    return popularity


# -------------------- Обучение ALS --------------------
def train_als(interactions_files, n_factors=64, reg=1e-3, device="cuda"):
    """
    Версия с сохранением батчей на указанный диск
    """
    # 1. ПРОХОД: Построение маппингов
    user_set = set()
    item_set = set()
    log_message("Первый проход: построение маппингов...")

    for f in tqdm(interactions_files):
        df = pl.read_parquet(f, columns=["user_id", "item_id"])
        user_set.update(df["user_id"].unique().to_list())
        item_set.update(df["item_id"].unique().to_list())

    user_map = {u: i for i, u in enumerate(sorted(user_set))}
    item_map = {i: j for j, i in enumerate(sorted(item_set))}
    log_message(
        f"Маппинги построены. Уников: users={len(user_map)}, items={len(item_map)}"
    )

    # 2. ПРОХОД: Сохранение батчей на указанный диск
    log_message("Сохранение батчей на диск...")

    # Используем указанный адрес вместо временной директории
    batch_dir = "/home/root6/python/e_cup/rec_system/data/processed/als_batches/"
    os.makedirs(batch_dir, exist_ok=True)

    user_map_df = pl.DataFrame(
        {"user_id": list(user_map.keys()), "user_idx": list(user_map.values())}
    )
    item_map_df = pl.DataFrame(
        {"item_id": list(item_map.keys()), "item_idx": list(item_map.values())}
    )

    batch_files = []
    for i, f in enumerate(tqdm(interactions_files)):
        df = pl.read_parquet(f, columns=["user_id", "item_id", "weight"])

        df = df.join(user_map_df, on="user_id", how="inner")
        df = df.join(item_map_df, on="item_id", how="inner")

        if len(df) > 0:
            # Сохраняем батч на указанный диск
            batch_path = os.path.join(batch_dir, f"batch_{i:04d}.npz")
            np.savez(
                batch_path,
                rows=df["user_idx"].to_numpy().astype(np.int32),
                cols=df["item_idx"].to_numpy().astype(np.int32),
                vals=df["weight"].to_numpy().astype(np.float32),
            )
            batch_files.append(batch_path)

    # 3. Постепенная загрузка и обучение
    log_message("Постепенное обучение...")

    als_model = TorchALS(
        len(user_map), len(item_map), n_factors=n_factors, device=device
    )

    for batch_path in tqdm(batch_files):
        try:
            # Загружаем батч с диска
            data = np.load(batch_path)
            rows = data["rows"]
            cols = data["cols"]
            vals = data["vals"]

            # Создаем sparse tensor для батча
            indices_np = np.empty((2, len(rows)), dtype=np.int32)
            indices_np[0] = rows
            indices_np[1] = cols
            indices = torch.tensor(indices_np, dtype=torch.long, device="cuda")
            values = torch.tensor(vals, dtype=torch.float32, device="cuda")
            sparse_batch = torch.sparse_coo_tensor(
                indices, values, size=(len(user_map), len(item_map)), device="cuda"
            )

            # Обучаем на одном батче
            als_model.partial_fit(sparse_batch, iterations=5, lr=0.005)

            # Очищаем
            del sparse_batch, indices, values
            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            log_message(f"Ошибка обработки батча {batch_path}: {e}")
            continue

    # Опционально: удаляем временные файлы после обучения
    log_message("Очистка временных файлов...")
    for batch_path in batch_files:
        try:
            os.remove(batch_path)
        except Exception as e:
            log_message(f"Ошибка удаления файла {batch_path}: {e}")

    # Проверяем, пуста ли директория и удаляем ее
    try:
        if not os.listdir(batch_dir):
            os.rmdir(batch_dir)
            log_message("Директория батчей удалена")
        else:
            log_message("В директории остались файлы, не удаляем")
    except Exception as e:
        log_message(f"Ошибка удаления директории: {e}")

    log_message("Обучение завершено!")
    return als_model, user_map, item_map


def build_copurchase_map(
    train_orders_df, min_co_items=2, top_n=20, device="cuda", max_items=1_000_000
):
    """
    строим словарь совместных покупок для топ-N товаров
    """
    log_message("Строим co-purchase матрицу для топ-N товаров...")

    # 1. Находим топ-10000 популярных товаров
    item_popularity = train_orders_df["item_id"].value_counts()
    top_items = item_popularity.head(max_items).index.tolist()
    popular_items_set = set(top_items)

    log_message(f"Топ-{len(top_items)} популярных товаров определены")

    # 2. Группируем корзины и фильтруем только популярные товары
    baskets = []
    for items in train_orders_df.groupby(["user_id", "created_timestamp"])[
        "item_id"
    ].apply(list):
        # Фильтруем только популярные товары
        filtered_items = [item for item in items if item in popular_items_set]
        if len(filtered_items) >= min_co_items:
            baskets.append(filtered_items)

    if not baskets:
        log_message("Нет корзин с популярными товарами")
        return {}

    log_message(f"Обрабатываем {len(baskets)} корзин с популярными товарами")

    # 3. Словарь {item_id -> index} только для популярных товаров
    item2idx = {it: i for i, it in enumerate(top_items)}
    idx2item = {i: it for it, i in item2idx.items()}
    n_items = len(top_items)

    log_message(f"Уникальных популярных товаров: {n_items}")

    # 4. Вместо плотной матрицы используем sparse coo format
    rows, cols, values = [], [], []

    for items in tqdm(baskets, desc="Обработка корзин"):
        if len(items) < min_co_items:
            continue

        # Получаем индексы товаров в корзине
        idxs = [item2idx[it] for it in items if it in item2idx]
        if len(idxs) < 2:
            continue

        weight = 1.0 / len(idxs)

        # Добавляем все пары товаров в корзине
        for i in range(len(idxs)):
            for j in range(len(idxs)):
                if i != j:  # исключаем диагональ
                    rows.append(idxs[i])
                    cols.append(idxs[j])
                    values.append(weight)

    # 5. Создаем sparse матрицу на GPU
    if not rows:
        log_message("Нет данных для построения матрицы")
        return {}

    log_message(f"Создаем sparse матрицу из {len(rows)} взаимодействий...")

    rows_tensor = torch.tensor(rows, dtype=torch.long, device=device)
    cols_tensor = torch.tensor(cols, dtype=torch.long, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)

    co_matrix = torch.sparse_coo_tensor(
        torch.stack([rows_tensor, cols_tensor]),
        values_tensor,
        size=(n_items, n_items),
        device=device,
    ).coalesce()  # объединяем дубликаты

    log_message(
        f"Sparse матрица построена: {co_matrix.shape}, ненулевых элементов: {co_matrix._nnz()}"
    )

    # 6. Нормализация построчно
    row_sums = torch.sparse.sum(co_matrix, dim=1).to_dense().clamp(min=1e-9)

    # 7. Формируем финальный словарь топ-N для каждого item
    final_copurchase = {}
    indices = co_matrix.indices()
    values = co_matrix.values()

    log_message("Формируем рекомендации...")
    for i in tqdm(range(n_items), desc="Обработка товаров"):
        # Находим все элементы в i-й строке
        mask = indices[0] == i
        if mask.any():
            col_indices = indices[1][mask]
            row_values = values[mask] / row_sums[i]  # нормализуем

            # Берем топ-N
            if len(row_values) > 0:
                topk_vals, topk_idx = torch.topk(
                    row_values, k=min(top_n, len(row_values))
                )
                final_copurchase[idx2item[i]] = [
                    (idx2item[col_indices[j].item()], topk_vals[j].item())
                    for j in range(len(topk_vals))
                    if topk_vals[j].item() > 0
                ]

    log_message(f"Co-purchase словарь построен для {len(final_copurchase)} товаров")

    # 8. Статистика
    avg_recommendations = sum(len(v) for v in final_copurchase.values()) / max(
        1, len(final_copurchase)
    )
    log_message(f"В среднем {avg_recommendations:.1f} рекомендаций на товар")

    return final_copurchase


def build_category_maps(items_df, categories_df):
    """
    Ускоренная версия: строим маппинги товаров и категорий.
    """
    log_message("Построение категорийных маппингов...")

    # Товар -> категория
    item_to_cat = dict(zip(items_df["item_id"], items_df["catalogid"]))

    # Категория -> список товаров
    cat_to_items = (
        items_df.groupby("catalogid")["item_id"].apply(lambda x: x.to_numpy()).to_dict()
    )

    # Иерархия категорий
    cat_tree = dict(zip(categories_df["catalogid"], categories_df["ids"]))

    # Расширение категорий через иерархию (векторизированно)
    extended_cat_to_items = {}
    for cat_id, items_list in cat_to_items.items():
        all_items = set(items_list)
        parents = cat_tree.get(cat_id, [])
        for parent in parents:
            if parent in cat_to_items:
                all_items.update(cat_to_items[parent])
        extended_cat_to_items[cat_id] = np.array(list(all_items))

    return item_to_cat, extended_cat_to_items


# -------------------- Метрики --------------------
def ndcg_at_k(recommended, ground_truth, k=100, device="cuda"):
    """
    NDCG@K: считаем через torch на GPU
    recommended: список рекомендованных item_id
    ground_truth: множество/список правильных item_id
    """
    if not ground_truth:
        return 0.0

    # Берём топ-k рекомендаций
    rec_k = torch.tensor(recommended[:k], device=device)

    # Множество правильных товаров
    gt_set = set(ground_truth)
    gt_mask = torch.tensor(
        [1 if x.item() in gt_set else 0 for x in rec_k],
        dtype=torch.float32,
        device=device,
    )

    # Позиции (1..k)
    positions = torch.arange(1, len(rec_k) + 1, device=device, dtype=torch.float32)

    # DCG: релевантность / log2(позиция+1)
    dcg = torch.sum(gt_mask / torch.log2(positions + 1))

    # IDCG: идеальный порядок
    ideal_len = min(len(ground_truth), k)
    idcg = torch.sum(
        1.0
        / torch.log2(
            torch.arange(1, ideal_len + 1, device=device, dtype=torch.float32) + 1
        )
    )

    return (dcg / idcg).item() if idcg > 0 else 0.0


def build_recent_items_map_from_batches(batch_dir, recent_n=5):
    """Версия где weight влияет на порядок items"""
    batch_files = sorted(Path(batch_dir).glob("*.parquet"))
    recent_items_map = {}

    for f in tqdm(batch_files, desc="Обработка батчей"):
        try:
            # Читаем с weight
            df = pl.read_parquet(
                f, columns=["user_id", "item_id", "timestamp", "weight"]
            )

            df = df.with_columns(
                [
                    pl.col("user_id").cast(pl.Int64),
                    pl.col("item_id").cast(pl.Int64),
                    pl.col("timestamp").dt.epoch("s").alias("ts_epoch"),
                    pl.col("weight").cast(pl.Float64),
                ]
            )

            # Создаем комбинированный score: weight + временной коэффициент
            # (новые взаимодействия имеют небольшое преимущество)
            df = df.with_columns(
                (
                    pl.col("weight") * 0.8
                    + pl.col("ts_epoch") / pl.col("ts_epoch").max() * 0.2
                ).alias("score")
            )

            # Сортируем по score
            df_sorted = df.sort(["user_id", "score"], descending=[False, True])

            # Группируем и берем топ-N
            grouped = df_sorted.group_by("user_id").agg(
                pl.col("item_id").head(recent_n).alias("items")
            )

            # Обновляем словарь (простая версия)
            for row in grouped.iter_rows():
                user_id, items = row[0], row[1]

                if user_id not in recent_items_map:
                    recent_items_map[user_id] = items
                else:
                    combined = (recent_items_map[user_id] + items)[:recent_n]
                    recent_items_map[user_id] = combined

        except Exception as e:
            log_message(f"Ошибка обработки файла {f}: {e}")
            continue

    return recent_items_map


# -------------------- Сохранение --------------------
def save_model(model, user_map, item_map, path="src/models/model_als.pkl"):
    """
    Сохраняет модель и маппинги в pickle.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {
        "model": model,
        "user_map": user_map,
        "item_map": item_map,
    }

    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    log_message(f"✅ Модель сохранена: {path}")


# -------------------- Метрики --------------------
def ndcg_at_k_grouped(predictions, targets, groups, k=100, device="cpu"):
    """
    Вычисление NDCG@k для сгруппированных данных (например, пользователей) с использованием PyTorch.

    predictions: 1D массив предсказанных score (list, np.array или torch.tensor)
    targets:     1D массив целевых значений (0/1) той же длины
    groups:      список размеров групп (например, сколько айтемов у каждого пользователя)
    k:           топ-K для метрики
    device:      "cpu" или "cuda"
    """
    preds = torch.as_tensor(predictions, dtype=torch.float32, device=device)
    targs = torch.as_tensor(targets, dtype=torch.float32, device=device)

    ndcg_scores = []
    start_idx = 0

    for group_size in groups:
        if group_size == 0:
            continue

        end_idx = start_idx + group_size
        group_preds = preds[start_idx:end_idx]
        group_targs = targs[start_idx:end_idx]

        # сортировка по убыванию предсказаний
        sorted_idx = torch.argsort(group_preds, descending=True)
        sorted_targs = group_targs[sorted_idx]

        # DCG
        denom = torch.log2(
            torch.arange(2, 2 + min(k, group_size), device=device, dtype=torch.float32)
        )
        dcg = (sorted_targs[:k] / denom).sum()

        # IDCG
        ideal_sorted = torch.sort(group_targs, descending=True).values
        idcg = (ideal_sorted[:k] / denom).sum()

        ndcg = (dcg / idcg) if idcg > 0 else torch.tensor(0.0, device=device)
        ndcg_scores.append(ndcg)

        start_idx = end_idx

    if not ndcg_scores:
        return 0.0

    return float(torch.stack(ndcg_scores).mean().cpu())


class TorchALS(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        n_factors=64,
        reg=1e-3,
        dtype=torch.float32,
        device="cuda",
    ):
        super().__init__()
        self.user_factors = nn.Parameter(
            nn.init.xavier_normal_(
                torch.empty(n_users, n_factors, dtype=dtype, device=device)
            )
        )
        self.item_factors = nn.Parameter(
            nn.init.xavier_normal_(
                torch.empty(n_items, n_factors, dtype=dtype, device=device)
            )
        )
        self.reg = reg
        self.device = device
        self.partial_optimizer = None  # Оптимизатор для partial_fit
        self.to(device)

    def forward(self, user, item):
        return (self.user_factors[user] * self.item_factors[item]).sum(1)

    def partial_fit(self, sparse_batch, iterations=5, lr=0.005, show_progress=False):
        """
        Инкрементальное обучение на ВСЕХ данных батча
        """
        # Подготавливаем sparse tensor
        if not sparse_batch.is_coalesced():
            sparse_batch = sparse_batch.coalesce()

        users_coo, items_coo = sparse_batch.indices()
        values = sparse_batch.values()

        # Инициализируем оптимизатор при первом вызове
        if self.partial_optimizer is None:
            self.partial_optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=self.reg
            )
        else:
            # Обновляем learning rate
            for param_group in self.partial_optimizer.param_groups:
                param_group["lr"] = lr

        # Обучаем на всех данных батча (без семплирования!)
        for epoch in range(iterations):
            self.partial_optimizer.zero_grad()

            # Предсказания для всех взаимодействий в батче
            pred = self.forward(users_coo, items_coo)
            loss = F.mse_loss(pred, values)

            # Регуляризация на всех данных
            user_reg = self.reg * self.user_factors[users_coo].pow(2).mean()
            item_reg = self.reg * self.item_factors[items_coo].pow(2).mean()
            total_loss = loss + user_reg + item_reg

            total_loss.backward()
            self.partial_optimizer.step()

            if show_progress and (epoch % 10 == 0 or epoch == iterations - 1):
                log_message(f"Partial fit epoch {epoch}, Loss: {total_loss.item():.6f}")


class LightGBMRecommender:
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.user_embeddings = None
        self.item_embeddings = None
        self.external_embeddings_dict = None
        self.copurchase_map = None
        self.item_to_cat = None
        self.cat_to_items = None
        self.user_map = None
        self.item_map = None
        self.covisitation_matrix = None

    def set_als_embeddings(self, als_model):
        """Сохраняем ALS эмбеддинги для использования в признаках"""
        self.user_embeddings = als_model.user_factors
        self.item_embeddings = als_model.item_factors

    def set_external_embeddings(self, embeddings_dict):
        """Устанавливаем внешние эмбеддинги товаров"""
        self.external_embeddings_dict = embeddings_dict

    def set_additional_data(
        self, copurchase_map, item_to_cat, cat_to_items, user_map, item_map
    ):
        """Устанавливаем дополнительные данные"""
        self.copurchase_map = copurchase_map
        self.item_to_cat = item_to_cat
        self.cat_to_items = cat_to_items
        self.user_map = user_map
        self.item_map = item_map

    def _train_covisitation_matrix(
        self, train_data: pd.DataFrame, min_cooccurrence: int = 5
    ):
        """Обучение матрицы ковизитации на тренировочных данных."""
        log_message("Обучение матрицы ковизитации на train данных...")

        # Создаем словарь: user_id -> список item_id с которыми взаимодействовал
        user_items = {}
        for user_id, group in train_data.groupby("user_id"):
            user_items[user_id] = set(group["item_id"].unique())

        # Считаем ковизитацию (совместные появления)
        cooccurrence = defaultdict(int)

        for user_id, items in user_items.items():
            items_list = list(items)
            for i in range(len(items_list)):
                for j in range(i + 1, len(items_list)):
                    item1, item2 = items_list[i], items_list[j]
                    # Упорядочиваем пару для consistency
                    pair = (min(item1, item2), max(item1, item2))
                    cooccurrence[pair] += 1

        # Фильтруем по минимальному количеству совместных появлений
        self.covisitation_matrix = {}
        for (item1, item2), count in cooccurrence.items():
            if count >= min_cooccurrence:
                self.covisitation_matrix[item1] = (
                    self.covisitation_matrix.get(item1, 0) + count
                )
                self.covisitation_matrix[item2] = (
                    self.covisitation_matrix.get(item2, 0) + count
                )

        log_message(
            f"Размер матрицы ковизитации: {len(self.covisitation_matrix)} товаров"
        )

    def _load_ui_features_for_pairs(self, pairs_df, ui_features_path):
        """
        Загружает UI-признаки только для указанных пар user-item
        """
        try:
            if pairs_df.empty:
                return None

            # Создаем временный файл с парами для фильтрации
            temp_pairs_path = "/tmp/filter_pairs.parquet"
            pairs_df[["user_id", "item_id"]].to_parquet(temp_pairs_path, index=False)

            # Используем Polars для эффективной фильтрации
            result = (
                pl.scan_parquet(ui_features_path)
                .join(
                    pl.scan_parquet(temp_pairs_path),
                    on=["user_id", "item_id"],
                    how="inner",
                )
                .collect()
                .to_pandas()
            )

            # Чистим временный файл
            if os.path.exists(temp_pairs_path):
                os.remove(temp_pairs_path)

            return result

        except Exception as e:
            log_message(f"Ошибка загрузки UI-признаков для пар: {e}")
            return None

    def _add_rich_features(
        self, data: pd.DataFrame, train_only_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Добавление расширенных фич с предотвращением утечки данных."""

        log_message("Добавление богатых признаков...")

        # Временные фичи
        if "timestamp" in data.columns:
            try:
                data["is_weekend"] = data["timestamp"].dt.dayofweek >= 5
                data["hour"] = data["timestamp"].dt.hour
            except Exception as e:
                log_message(f"Не удалось преобразовать timestamp: {e}")
                data["is_weekend"] = 0
                data["hour"] = -1
        else:
            data["is_weekend"] = np.nan
            data["hour"] = np.nan

        # Если train_only_data не передан, берём подмножество data
        if train_only_data is None or train_only_data.empty:
            train_only_data = data[data.get("target", 0) == 0]

        # --- Популярность товара ---
        if not train_only_data.empty:
            item_pop = (
                train_only_data.groupby("item_id")["user_id"]
                .count()
                .rename("item_popularity")
                .reset_index()
            )
            # Проверяем, что item_pop не пустой
            if not item_pop.empty:
                data = data.merge(item_pop, on="item_id", how="left")

        # Создаем колонку если ее нет или заполняем нулями
        if "item_popularity" not in data.columns:
            data["item_popularity"] = 0
        else:
            data["item_popularity"] = data["item_popularity"].fillna(0)

        # --- Популярность категории ---
        if "category_id" in data.columns:
            if not train_only_data.empty and "category_id" in train_only_data.columns:
                cat_pop = (
                    train_only_data.groupby("category_id")["user_id"]
                    .count()
                    .rename("category_popularity")
                    .reset_index()
                )
                if not cat_pop.empty:
                    data = data.merge(cat_pop, on="category_id", how="left")

            if "category_popularity" not in data.columns:
                data["category_popularity"] = 0
            else:
                data["category_popularity"] = data["category_popularity"].fillna(0)
        else:
            data["category_popularity"] = 0

        # --- Активность пользователя ---
        if not train_only_data.empty:
            user_activity = (
                train_only_data.groupby("user_id")["item_id"]
                .count()
                .rename("user_activity")
                .reset_index()
            )
            if not user_activity.empty:
                data = data.merge(user_activity, on="user_id", how="left")

        if "user_activity" not in data.columns:
            data["user_activity"] = 0
        else:
            data["user_activity"] = data["user_activity"].fillna(0)

        # --- Средняя популярность товаров у пользователя ---
        if not train_only_data.empty and "item_popularity" in data.columns:
            train_with_pop = (
                train_only_data.merge(item_pop, on="item_id", how="left")
                if not item_pop.empty
                else train_only_data.copy()
            )

            if "item_popularity" in train_with_pop.columns:
                train_with_pop["item_popularity"] = train_with_pop[
                    "item_popularity"
                ].fillna(0)
                user_avg_pop = (
                    train_with_pop.groupby("user_id")["item_popularity"]
                    .mean()
                    .rename("user_avg_item_popularity")
                    .reset_index()
                )
                if not user_avg_pop.empty:
                    data = data.merge(user_avg_pop, on="user_id", how="left")

        if "user_avg_item_popularity" not in data.columns:
            data["user_avg_item_popularity"] = 0
        else:
            data["user_avg_item_popularity"] = data["user_avg_item_popularity"].fillna(
                0
            )

        # --- Ковизитация ---
        if (
            hasattr(self, "covisitation_matrix")
            and self.covisitation_matrix is not None
        ):
            data["covisitation_score"] = (
                data["item_id"]
                .map(self.covisitation_matrix.get, na_action="ignore")
                .fillna(0)
            )
        else:
            data["covisitation_score"] = 0

        # --- FCLIP эмбеддинги ---
        if getattr(self, "external_embeddings_dict", None):
            log_message("Ускоренная обработка FCLIP эмбеддингов на GPU...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            all_item_ids = list(self.external_embeddings_dict.keys())
            embedding_dim = len(next(iter(self.external_embeddings_dict.values())))
            n_fclip_dims = min(10, embedding_dim)

            embeddings_tensor = torch.zeros(
                len(all_item_ids), embedding_dim, device=device
            )
            for idx, item_id in enumerate(all_item_ids):
                embeddings_tensor[idx] = torch.tensor(
                    self.external_embeddings_dict[item_id],
                    device=device,
                    dtype=torch.float32,
                )

            item_id_to_idx = {item_id: idx for idx, item_id in enumerate(all_item_ids)}
            batch_size = 100_000
            total_rows = len(data)

            for i in range(n_fclip_dims):
                log_message(f"Обработка FCLIP измерения {i+1}/{n_fclip_dims} на GPU...")
                data[f"fclip_embed_{i}"] = 0.0

                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch_data = data.iloc[start_idx:end_idx]
                    batch_item_ids = batch_data["item_id"].values
                    valid_mask = np.array(
                        [item_id in item_id_to_idx for item_id in batch_item_ids]
                    )
                    valid_indices = np.where(valid_mask)[0]
                    valid_item_ids = batch_item_ids[valid_mask]

                    if len(valid_item_ids) > 0:
                        tensor_indices = torch.tensor(
                            [item_id_to_idx[item_id] for item_id in valid_item_ids],
                            device=device,
                        )
                        batch_embeddings = (
                            embeddings_tensor[tensor_indices, i].cpu().numpy()
                        )
                        data.iloc[
                            start_idx + valid_indices,
                            data.columns.get_loc(f"fclip_embed_{i}"),
                        ] = batch_embeddings

                    del batch_data, batch_item_ids
                    if start_idx % (batch_size * 5) == 0:
                        torch.cuda.empty_cache()

            del embeddings_tensor, item_id_to_idx
            torch.cuda.empty_cache()

        # --- Регистрируем новые признаки ---
        new_features = [
            "is_weekend",
            "hour",
            "item_popularity",
            "category_popularity",
            "user_activity",
            "user_avg_item_popularity",
            "covisitation_score",
        ]
        if getattr(self, "external_embeddings_dict", None):
            new_features += [f"fclip_embed_{i}" for i in range(n_fclip_dims)]

        existing_features = set(getattr(self, "feature_columns", []))
        for feature in new_features:
            if feature in data.columns and feature not in existing_features:
                self.feature_columns.append(feature)
                existing_features.add(feature)

        log_message(f"Добавлены фичи: {[f for f in new_features if f in data.columns]}")
        log_message(f"Всего фич после добавления: {len(self.feature_columns)}")

        return data

    def prepare_training_data(
        self,
        interactions_files,
        orders_ddf,
        user_map,
        item_map,
        popularity_s,
        recent_items_map,
        sample_fraction=0.1,
        negatives_per_positive=2,
        ui_features_dir=None,
        val_days: int = 7,
    ):
        log_message("Подготовка данных для LightGBM (streaming)...")

        # берем только fraction каждой партиции
        orders_ddf = orders_ddf.map_partitions(
            lambda df: df.sample(frac=sample_fraction, random_state=42)
        )

        base_dir = Path("/home/root6/python/e_cup/rec_system/data/processed/")
        base_dir.mkdir(parents=True, exist_ok=True)
        train_path = base_dir / "train.parquet"
        val_path = base_dir / "val.parquet"

        batch_size = 5000  # маленькие батчи, чтобы память не перегружать

        # --- Загружаем взаимодействия ---
        interactions_chunks = []
        for f in tqdm(interactions_files, desc="Загрузка взаимодействий"):
            df = pd.read_parquet(
                f, columns=["user_id", "item_id", "timestamp", "weight"]
            )
            df["user_id"] = df["user_id"].astype("int64")
            df["item_id"] = df["item_id"].astype("int64")
            interactions_chunks.append(df)
        interactions_df = pd.concat(interactions_chunks, ignore_index=True)
        interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
        max_timestamp = interactions_df["timestamp"].max()

        split_time = max_timestamp - pd.Timedelta(days=val_days)
        train_timestamp_fill = split_time - pd.Timedelta(seconds=1)

        # ВАЖНО: Разделяем interactions на train и val для user_interacted_items
        interactions_train = interactions_df[interactions_df["timestamp"] <= split_time]
        interactions_val = interactions_df[interactions_df["timestamp"] > split_time]

        # Для генерации train используем только train interactions
        user_interacted_items_train = {
            uid: set(gr["item_id"].unique())
            for uid, gr in interactions_train.groupby("user_id")
        }

        all_items = np.array(list(item_map.keys()), dtype=np.int64)
        popular_items_set = set(popularity_s.nlargest(10000).index.astype(np.int64))

        cols_order = ["user_id", "item_id", "timestamp", "target"]

        # --- Инициализация ParquetWriter ---
        writer_train = None
        writer_val = None

        def write_small_batch(batch_list):
            nonlocal writer_train, writer_val
            batch_df = pd.DataFrame(batch_list, columns=cols_order)
            train_df = batch_df[batch_df["timestamp"] <= split_time]
            val_df = batch_df[batch_df["timestamp"] > split_time]

            if not train_df.empty:
                table_train = pa.Table.from_pandas(train_df, preserve_index=False)
                if writer_train is None:
                    writer_train = pq.ParquetWriter(
                        train_path, schema=table_train.schema
                    )
                writer_train.write_table(table_train)

            if not val_df.empty:
                table_val = pa.Table.from_pandas(val_df, preserve_index=False)
                if writer_val is None:
                    writer_val = pq.ParquetWriter(val_path, schema=table_val.schema)
                writer_val.write_table(table_val)

        small_batch = []

        # --- Генерация TRAIN данных ---
        for part in tqdm(orders_ddf.to_delayed(), desc="Streaming генерация"):
            pdf = part.compute()  # только одна партиция в память
            for uid, gr in pdf.groupby("user_id"):
                pos_items = set(gr[gr["target"] == 1]["item_id"])
                neg_items_existing = set(gr[gr["target"] == 0]["item_id"])
                # Используем только train interactions для исключения
                interacted = user_interacted_items_train.get(uid, set())
                excluded = pos_items | neg_items_existing | interacted
                available_items = all_items[~np.isin(all_items, list(excluded))]

                # Позитивы
                for it in pos_items:
                    small_batch.append([uid, it, train_timestamp_fill, 1])

                # Существующие негативы
                for it in neg_items_existing:
                    small_batch.append([uid, it, train_timestamp_fill, 0])

                # Дополнительные негативы
                n_needed = max(
                    0, len(pos_items) * negatives_per_positive - len(neg_items_existing)
                )
                if n_needed > 0 and len(available_items) > 0:
                    popular_mask = np.isin(available_items, list(popular_items_set))
                    popular_candidates = available_items[popular_mask]
                    random_candidates = available_items[~popular_mask]

                    n_popular = min(n_needed // 2, len(popular_candidates))
                    n_random = min(n_needed - n_popular, len(random_candidates))

                    sampled_items = []
                    if n_popular > 0:
                        sampled_items.extend(
                            np.random.choice(
                                popular_candidates, n_popular, replace=False
                            )
                        )
                    if n_random > 0:
                        sampled_items.extend(
                            np.random.choice(random_candidates, n_random, replace=False)
                        )

                    for it in sampled_items:
                        small_batch.append([uid, it, train_timestamp_fill, 0])

                # --- Если набрался небольшой батч, пишем ---
                if len(small_batch) >= batch_size:
                    write_small_batch(small_batch)
                    small_batch = []

        # --- Записываем остаток ---
        if small_batch:
            write_small_batch(small_batch)

        # --- Закрываем writers ---
        if writer_train:
            writer_train.close()
        if writer_val:
            writer_val.close()
        else:
            empty_val = pd.DataFrame(columns=cols_order)
            empty_val.to_parquet(val_path, engine="pyarrow", index=False)

        # --- Загружаем train/val ---
        train_data = pd.read_parquet(train_path)
        val_data = pd.read_parquet(val_path)
        if len(val_data) == 0:
            cutoff = int(len(train_data) * 0.9)
            train_data, val_data = (
                train_data.iloc[:cutoff].copy(),
                train_data.iloc[cutoff:].copy(),
            )

        # ВАЖНО: Обучаем ковизитацию на train данных ДО добавления фич
        self._train_covisitation_matrix(train_data)

        # Теперь добавляем фичи
        log_message("Добавление фичей для train данных...")
        train_data = self._add_rich_features(train_data, train_only_data=train_data)

        log_message("Добавление фичей для val данных (на основе train статистик)...")
        val_data = self._add_rich_features(val_data, train_only_data=train_data)

        if ui_features_dir and os.path.exists(ui_features_dir):
            train_data, val_data = self._add_ui_features_optimized(
                train_data, val_data, ui_features_dir
            )

        # --- Лог ---
        def log_message_dist(df, name):
            counts = df["target"].value_counts(dropna=False).to_dict()
            log_message(f"{name}: rows={len(df)}; target_counts={counts}")

        log_message_dist(train_data, "TRAIN")
        log_message_dist(val_data, "VAL")
        log_message(f"split_time = {split_time} (val_days={val_days})")
        log_message(
            f"✅ Данные подготовлены: train={len(train_data)}, val={len(val_data)}"
        )

        return train_data, val_data

    # Добавляем оптимизированный метод для UI фич
    def _add_ui_features_optimized(self, train_data, val_data, ui_features_dir):
        """Оптимизированное добавление UI фич"""
        try:
            # Объединяем данные для batch processing
            all_data = pd.concat(
                [train_data[["user_id", "item_id"]], val_data[["user_id", "item_id"]]],
                ignore_index=True,
            )

            ui_features_batch = get_ui_features_batch(
                all_data.to_dict("records"), ui_features_dir
            )

            if ui_features_batch:
                ui_features_df = pd.DataFrame(ui_features_batch)
                ui_features_df["user_id"] = ui_features_df["user_id"].astype("int64")
                ui_features_df["item_id"] = ui_features_df["item_id"].astype("int64")

                # Убираем дубликаты
                ui_features_df = ui_features_df.drop_duplicates(
                    subset=["user_id", "item_id"]
                )

                # Объединяем с train и val
                train_data = train_data.merge(
                    ui_features_df, on=["user_id", "item_id"], how="left"
                ).fillna(0)
                val_data = val_data.merge(
                    ui_features_df, on=["user_id", "item_id"], how="left"
                ).fillna(0)

                # Обновляем feature_columns
                new_features = [
                    col
                    for col in ui_features_df.columns
                    if col not in ["user_id", "item_id"]
                    and col not in self.feature_columns
                ]
                self.feature_columns.extend(new_features)

                log_message(f"Добавлены UI признаки: {new_features}")

        except Exception as e:
            log_message(f"Ошибка загрузки UI-признаков: {e}")

        return train_data, val_data

    def _get_copurchase_strength(self, item_id):
        """Получаем силу co-purchase связи"""
        if not self.copurchase_map or item_id not in self.copurchase_map:
            return 0.0

        # Максимальная сила связи с этим товаром
        strengths = [strength for _, strength in self.copurchase_map[item_id]]
        return max(strengths) if strengths else 0.0

    def _get_user_copurchase_affinity(self, user_id, item_id):
        """Affinity пользователя к co-purchase связям"""
        if not self.copurchase_map or not hasattr(self, "user_items_history"):
            return 0.0

        # Здесь нужно добавить логику расчета на основе истории пользователя
        # Временная реализация - возвращаем общую силу
        return self._get_copurchase_strength(item_id)

    def train(self, train_data, val_data=None, params=None):
        """
        Обучение LightGBM с NDCG optimization
        """
        if params is None:
            # params = {
            #     "objective": "lambdarank",
            #     "metric": "ndcg",
            #     "ndcg_eval_at": [100],
            #     "learning_rate": 0.05,
            #     "num_leaves": 31,
            #     "max_depth": -1,
            #     "min_data_in_leaf": 20,  # Увеличить для предотвращения переобучения
            #     "feature_fraction": 0.8,
            #     "bagging_fraction": 0.8,
            #     "bagging_freq": 5,
            #     "lambda_l1": 0.1,  # L1 регуляризация
            #     "lambda_l2": 0.1,  # L2 регуляризация
            #     "early_stopping_rounds": 10,
            # }
            params = {
                "objective": "binary",
                "metric": "binary_error",
                "learning_rate": 0.05,
                "num_leaves": 7,  # Очень простая модель
                "max_depth": 3,
            }

        # Группы для lambdarank (количество товаров на пользователя)
        train_groups = train_data.groupby("user_id").size().values
        X_train = train_data[self.feature_columns]
        y_train = train_data["target"]

        log_message(f"Размер тренировочных данных: {len(X_train)}")
        log_message(f"Количество групп: {len(train_groups)}")

        train_dataset = lgb.Dataset(
            X_train,
            label=y_train,
            group=train_groups,
            feature_name=list(X_train.columns),
        )

        if val_data is not None:
            val_groups = val_data.groupby("user_id").size().values
            X_val = val_data[self.feature_columns]
            y_val = val_data["target"]

            val_dataset = lgb.Dataset(
                X_val, label=y_val, group=val_groups, reference=train_dataset
            )
            valid_sets = [train_dataset, val_dataset]
            valid_names = ["train", "valid"]
        else:
            valid_sets = [train_dataset]
            valid_names = ["train"]

        log_message("Начинаем обучение LightGBM...")
        self.model = lgb.train(
            params,
            train_dataset,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.log_evaluation(50),
                lgb.early_stopping(5),
            ],
        )

        return self.model

    def evaluate(self, data):
        """Оценка модели"""
        if self.model is None:
            log_message("Модель не обучена")
            return 0.0

        if len(data) == 0:
            log_message("Нет данных для оценки")
            return 0.0

        predictions = self.predict_rank(data)
        groups = data.groupby("user_id").size().values
        ndcg = ndcg_at_k_grouped(predictions, data["target"].values, groups, k=100)
        return ndcg

    def predict_rank(self, data):
        """Предсказание рангов"""
        X = data[self.feature_columns]
        return self.model.predict(X)

    def recommend(self, user_items_data):
        """Генерация рекомендаций"""
        predictions = self.predict_rank(user_items_data)
        user_items_data["score"] = predictions

        recommendations = {}
        for user_id, group in user_items_data.groupby("user_id"):
            top_items = group.nlargest(100, "score")["item_id"].tolist()
            recommendations[user_id] = top_items

        return recommendations


def build_user_features_dict(interactions_files, orders_df, device="cuda"):
    """
    Оптимизированная версия с использованием Polars
    """
    import polars as pl

    log_message("Построение словаря пользовательских признаков...")

    # 1. АГРЕГАЦИЯ ПО ТРЕКЕРУ (взаимодействия)
    user_stats_list = []

    for f in tqdm(interactions_files, desc="Обработка трекера"):
        df = pl.read_parquet(f)

        chunk_stats = df.group_by("user_id").agg(
            [
                pl.col("weight").count().alias("count"),
                pl.col("weight").sum().alias("sum"),
                pl.col("weight").max().alias("max"),
                pl.col("weight").min().alias("min"),
                pl.col("timestamp").max().alias("last_ts"),
                pl.col("timestamp").min().alias("first_ts"),
            ]
        )

        user_stats_list.append(chunk_stats)

    # Объединяем все статистики
    if user_stats_list:
        all_stats = pl.concat(user_stats_list)
        final_stats = all_stats.group_by("user_id").agg(
            [
                pl.col("count").sum().alias("user_count"),
                pl.col("sum").sum().alias("user_sum"),
                pl.col("max").max().alias("user_max"),
                pl.col("min").min().alias("user_min"),
                pl.col("last_ts").max().alias("user_last_ts"),
                pl.col("first_ts").min().alias("user_first_ts"),
            ]
        )
    else:
        final_stats = pl.DataFrame()

    # 2. АГРЕГАЦИЯ ПО ЗАКАЗАМ
    log_message("Агрегация по заказам...")
    if isinstance(orders_df, pl.DataFrame):
        orders_pl = orders_df
    else:
        orders_pl = pl.from_pandas(
            orders_df.compute() if hasattr(orders_df, "compute") else orders_df
        )

    order_stats = orders_pl.group_by("user_id").agg(
        [
            pl.col("item_id").count().alias("user_orders_count"),
            pl.col("created_timestamp").max().alias("user_last_order_ts"),
            pl.col("created_timestamp").min().alias("user_first_order_ts"),
        ]
    )

    # 3. ОБЪЕДИНЕНИЕ ДАННЫХ
    if len(final_stats) > 0 and len(order_stats) > 0:
        user_stats = final_stats.join(
            order_stats, on="user_id", how="full"
        )  # Исправлено: outer -> full
    elif len(final_stats) > 0:
        user_stats = final_stats
    else:
        user_stats = order_stats

    # 4. ВЫЧИСЛЕНИЕ ПРОИЗВОДНЫХ ПРИЗНАКОВ
    current_time = pl.lit(datetime.now())  # Исправлено: pl.datetime.now() -> pl.now()

    user_stats = user_stats.with_columns(
        [
            # Заполняем пропущенные значения
            pl.col("user_count").fill_null(0),
            pl.col("user_sum").fill_null(0),
            pl.col("user_orders_count").fill_null(0),
            # Вычисляем среднее
            (pl.col("user_sum") / pl.col("user_count")).alias("user_mean"),
            # Время с последнего взаимодействия
            ((current_time - pl.col("user_last_ts")).dt.total_days()).alias(
                "user_days_since_last"
            ),
            # Время с первого взаимодействия
            ((current_time - pl.col("user_first_ts")).dt.total_days()).alias(
                "user_days_since_first"
            ),
            # Время с последнего заказа
            ((current_time - pl.col("user_last_order_ts")).dt.total_days()).alias(
                "user_days_since_last_order"
            ),
        ]
    ).fill_nan(
        0
    )  # Заполняем NaN от деления на 0

    # 5. КОНВЕРТАЦИЯ В СЛОВАРЬ
    user_stats_dict = {}
    for row in user_stats.iter_rows(named=True):
        user_stats_dict[row["user_id"]] = {
            "user_count": row["user_count"],
            "user_mean": row["user_mean"],
            "user_sum": row["user_sum"],
            "user_max": row["user_max"],
            "user_min": row["user_min"],
            "user_last_ts": row["user_last_ts"],
            "user_first_ts": row["user_first_ts"],
            "user_orders_count": row["user_orders_count"],
            "user_last_order_ts": row["user_last_order_ts"],
            "user_first_order_ts": row["user_first_order_ts"],
            "user_days_since_last": row["user_days_since_last"],
            "user_days_since_first": row["user_days_since_first"],
            "user_days_since_last_order": row["user_days_since_last_order"],
        }

    log_message(
        f"Словарь пользовательских признаков построен. Записей: {len(user_stats_dict)}"
    )
    return user_stats_dict


def load_ui_features_for_user_item(user_id, item_id, ui_features_path):
    """
    Загружает UI-признаки для конкретной пары user-item
    """
    if not ui_features_path or not os.path.exists(ui_features_path):
        return None

    query = pl.scan_parquet(ui_features_path).filter(
        (pl.col("user_id") == user_id) & (pl.col("item_id") == item_id)
    )
    result = query.collect()

    if len(result) == 0:
        return None

    return result[0].to_dict()


def build_item_features_dict(
    interactions_files, items_df, orders_df, embeddings_dict, device="cuda"
):
    """
    Оптимизированная версия с использованием Polars
    """
    import polars as pl

    log_message("Построение словаря товарных признаков...")

    # 1. АГРЕГАЦИЯ ПО ТРЕКЕРУ И ЗАКАЗАМ
    item_stats_list = []

    for f in tqdm(interactions_files, desc="Обработка взаимодействий"):
        df = pl.read_parquet(f)

        chunk_stats = df.group_by("item_id").agg(
            [
                pl.col("weight").count().alias("count"),
                pl.col("weight").sum().alias("sum"),
                pl.col("weight").max().alias("max"),
                pl.col("weight").min().alias("min"),
                pl.col("timestamp").max().alias("last_ts"),
                pl.col("timestamp").min().alias("first_ts"),
            ]
        )

        item_stats_list.append(chunk_stats)

    # Объединяем статистики
    if item_stats_list:
        all_stats = pl.concat(item_stats_list)
        final_stats = all_stats.group_by("item_id").agg(
            [
                pl.col("count").sum().alias("item_count"),
                pl.col("sum").sum().alias("item_sum"),
                pl.col("max").max().alias("item_max"),
                pl.col("min").min().alias("item_min"),
                pl.col("last_ts").max().alias("item_last_ts"),
                pl.col("first_ts").min().alias("item_first_ts"),
            ]
        )
    else:
        final_stats = pl.DataFrame()

    # 2. ДОБАВЛЕНИЕ ДАННЫХ ИЗ ЗАКАЗОВ
    if isinstance(orders_df, pl.DataFrame):
        orders_pl = orders_df
    else:
        orders_pl = pl.from_pandas(
            orders_df.compute() if hasattr(orders_df, "compute") else orders_df
        )

    order_stats = orders_pl.group_by("item_id").agg(
        [pl.col("user_id").count().alias("item_orders_count")]
    )

    # 3. ДОБАВЛЕНИЕ ДАННЫХ ИЗ items_df
    log_message("Добавление данных из items_df...")
    if isinstance(items_df, pl.DataFrame):
        items_pl = items_df
    else:
        items_pl = pl.from_pandas(
            items_df.compute() if hasattr(items_df, "compute") else items_df
        )

    items_catalog = items_pl.select(["item_id", "catalogid"]).unique()

    # 4. ОБЪЕДИНЕНИЕ ВСЕХ ДАННЫХ
    item_stats = final_stats.join(order_stats, on="item_id", how="full")  # Исправлено
    item_stats = item_stats.join(items_catalog, on="item_id", how="left")

    # 5. ВЫЧИСЛЕНИЕ ПРОИЗВОДНЫХ ПРИЗНАКОВ
    current_time = pl.lit(datetime.now())  # Исправлено

    item_stats = item_stats.with_columns(
        [
            pl.col("item_count").fill_null(0),
            pl.col("item_sum").fill_null(0),
            pl.col("item_orders_count").fill_null(0),
            (pl.col("item_sum") / pl.col("item_count")).alias("item_mean"),
            ((current_time - pl.col("item_last_ts")).dt.total_days()).alias(
                "item_days_since_last"
            ),
            ((current_time - pl.col("item_first_ts")).dt.total_days()).alias(
                "item_days_since_first"
            ),
        ]
    ).fill_nan(0)

    # 6. КОНВЕРТАЦИЯ В СЛОВАРЬ
    item_stats_dict = {}
    for row in item_stats.iter_rows(named=True):
        item_stats_dict[row["item_id"]] = {
            "item_count": row["item_count"],
            "item_mean": row["item_mean"],
            "item_sum": row["item_sum"],
            "item_max": row["item_max"],
            "item_min": row["item_min"],
            "item_last_ts": row["item_last_ts"],
            "item_first_ts": row["item_first_ts"],
            "item_orders_count": row["item_orders_count"],
            "item_category": row["catalogid"],
            "item_days_since_last": row["item_days_since_last"],
            "item_days_since_first": row["item_days_since_first"],
        }

    # 7. ДОБАВЛЕНИЕ ЭМБЕДДИНГОВ
    log_message("Добавление эмбеддингов...")
    for item_id, embedding in embeddings_dict.items():
        if item_id in item_stats_dict:
            for i in range(min(5, len(embedding))):
                item_stats_dict[item_id][f"fclip_embed_{i}"] = float(embedding[i])

    log_message(f"Словарь товарных признаков построен. Записей: {len(item_stats_dict)}")
    return item_stats_dict


def get_ui_features_for_user_item(user_id, item_id, ui_features_dir):
    """
    Ищет UI-признаки для пары user-item по всем файлам
    """
    try:
        metadata_path = os.path.join(ui_features_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return None

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Ищем в каждом файле
        for ui_file in metadata["ui_feature_files"]:
            if not os.path.exists(ui_file):
                continue

            result = (
                pl.scan_parquet(ui_file)
                .filter((pl.col("user_id") == user_id) & (pl.col("item_id") == item_id))
                .collect()
            )

            if not result.is_empty():
                return result[0].to_dict()

        return None

    except Exception as e:
        log_message(f"Ошибка поиска UI-признаков: {e}")
        return None


def get_ui_features_batch(user_item_pairs, ui_features_dir, batch_size=1000):
    """
    Получает UI-признаки для батча пар из всех файлов
    """
    try:

        metadata_path = os.path.join(ui_features_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return []

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        all_results = []

        # Обрабатываем каждый файл UI-признаков
        for ui_file in metadata["ui_feature_files"]:
            if not os.path.exists(ui_file):
                continue

            # Создаем временный файл с парами для фильтрации
            temp_pairs_path = "/tmp/filter_pairs.parquet"
            pairs_df = pl.DataFrame(user_item_pairs, schema=["user_id", "item_id"])
            pairs_df.write_parquet(temp_pairs_path)

            # Ищем совпадения в текущем файле
            results = (
                pl.scan_parquet(ui_file)
                .join(
                    pl.scan_parquet(temp_pairs_path),
                    on=["user_id", "item_id"],
                    how="inner",
                )
                .collect()
                .to_dicts()
            )

            all_results.extend(results)

            # Удаляем временный файл
            if os.path.exists(temp_pairs_path):
                os.remove(temp_pairs_path)

        return all_results

    except Exception as e:
        log_message(f"Ошибка получения батча UI-признаков: {e}")
        return []


def build_user_item_features_dict(
    interactions_files,
    output_dir="/home/root6/python/e_cup/rec_system/data/processed/ui_features",
    cleanup=False,
):
    """
    Создает отдельные файлы UI-признаков для каждого файла взаимодействий.
    Все колонки преобразованы в числовой тип для LightGBM.
    """
    import json
    import os
    from datetime import datetime

    import polars as pl
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)
    log_message("Создание UI-признаков (данные остаются на диске)")

    try:
        ui_feature_files = []

        for input_file in tqdm(interactions_files, desc="Создание UI-признаков"):
            try:
                input_filename = os.path.basename(input_file)
                output_file = os.path.join(output_dir, f"ui_features_{input_filename}")

                # Поларс агрегирует UI-признаки
                df = (
                    pl.scan_parquet(input_file)
                    .group_by(["user_id", "item_id"])
                    .agg(
                        [
                            pl.col("weight").count().alias("ui_count"),
                            pl.col("weight").sum().alias("ui_sum"),
                            pl.col("weight").max().alias("ui_max"),
                            pl.col("weight").min().alias("ui_min"),
                            pl.col("timestamp").max().alias("ui_last_ts"),
                            pl.col("timestamp").min().alias("ui_first_ts"),
                        ]
                    )
                    .with_columns(
                        [
                            # Среднее значение
                            (pl.col("ui_sum") / pl.col("ui_count"))
                            .fill_nan(0)
                            .alias("ui_mean"),
                            # Дни с последнего взаимодействия
                            (
                                (
                                    pl.lit(datetime.now()).cast(pl.Datetime)
                                    - pl.col("ui_last_ts").cast(pl.Datetime)
                                )
                                .dt.total_days()  # ← ИСПРАВЛЕНО: .days() → .total_days()
                                .cast(pl.Float64)
                            ).alias("ui_days_since_last"),
                            # Дни с первого взаимодействия
                            (
                                (
                                    pl.lit(datetime.now()).cast(pl.Datetime)
                                    - pl.col("ui_first_ts").cast(pl.Datetime)
                                )
                                .dt.total_days()  # ← ИСПРАВЛЕНО: .days() → .total_days()
                                .cast(pl.Float64)
                            ).alias("ui_days_since_first"),
                        ]
                    )
                    .select(
                        [
                            "user_id",
                            "item_id",
                            "ui_count",
                            "ui_sum",
                            "ui_max",
                            "ui_min",
                            "ui_mean",
                            "ui_days_since_last",
                            "ui_days_since_first",
                        ]
                    )
                    .fill_null(0)
                )

                df.sink_parquet(output_file)
                ui_feature_files.append(output_file)

            except Exception as e:
                log_message(f"Ошибка обработки файла {input_file}: {e}")
                continue

        # Сохраняем metadata
        metadata = {
            "created_date": datetime.now().isoformat(),
            "source_files": len(interactions_files),
            "ui_feature_files": ui_feature_files,
            "output_dir": output_dir,
        }
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        log_message(
            f"Создано {len(ui_feature_files)} файлов UI-признаков в: {output_dir}"
        )
        return output_dir

    except Exception as e:
        log_message(f"❌ Ошибка: {e}")
        import traceback

        traceback.log_message_exc()
        return None


def build_category_features_dict(category_df, items_df):
    """
    Оптимизированная версия с использованием Polars
    """
    import polars as pl

    log_message("Построение категорийных признаков...")

    if not isinstance(category_df, pl.DataFrame):
        category_pl = pl.from_pandas(
            category_df.compute() if hasattr(category_df, "compute") else category_df
        )
    else:
        category_pl = category_df

    if not isinstance(items_df, pl.DataFrame):
        items_pl = pl.from_pandas(
            items_df.compute() if hasattr(items_df, "compute") else items_df
        )
    else:
        items_pl = items_df

    # Создаем маппинг категория -> уровень в иерархии
    cat_levels = category_pl.with_columns(
        [(pl.col("ids").list.lengths() - 1).alias("category_level")]
    ).select(["catalogid", "category_level"])

    # Создаем маппинг товар -> категория
    item_categories = items_pl.select(["item_id", "catalogid"]).unique()

    # Объединяем
    category_features = item_categories.join(cat_levels, on="catalogid", how="left")
    category_features = category_features.with_columns(
        [pl.col("category_level").fill_null(0)]
    )

    # КОНВЕРТАЦИЯ В СЛОВАРЬ
    category_features_dict = {}
    for row in category_features.iter_rows(named=True):
        category_features_dict[row["item_id"]] = {
            "item_category": row["catalogid"],
            "category_level": row["category_level"],
        }

    log_message(
        f"Категорийные признаки построены. Записей: {len(category_features_dict)}"
    )
    return category_features_dict


def prepare_lgbm_training_data(
    user_features_dict,
    item_features_dict,
    user_item_features_dict,
    category_features_dict,
    test_orders_df,
    all_items,  # список всех item_id
    sample_fraction=0.1,
):
    """
    Подготавливает данные для обучения LightGBM:
    1 положительный пример -> 1 негативный пример.
    """
    log_message("Подготовка данных для обучения LightGBM...")

    # Берем sample от тестовых заказов
    test_sample = test_orders_df.sample(frac=sample_fraction, random_state=42)

    train_examples = []

    for _, row in test_sample.iterrows():
        user_id = row["user_id"]
        pos_item_id = row["item_id"]

        # ---------- Положительный ----------
        features = {}
        if user_id in user_features_dict:
            features.update(user_features_dict[user_id])
        if pos_item_id in item_features_dict:
            features.update(item_features_dict[pos_item_id])
        if (user_id, pos_item_id) in user_item_features_dict:
            features.update(user_item_features_dict[(user_id, pos_item_id)])
        if pos_item_id in category_features_dict:
            features.update(category_features_dict[pos_item_id])

        features["target"] = 1
        features["user_id"] = user_id
        features["item_id"] = pos_item_id
        train_examples.append(features)

        # ---------- Негативный ----------
        neg_item_id = random.choice(all_items)
        while neg_item_id == pos_item_id:
            neg_item_id = random.choice(all_items)

        neg_features = {}
        if user_id in user_features_dict:
            neg_features.update(user_features_dict[user_id])
        if neg_item_id in item_features_dict:
            neg_features.update(item_features_dict[neg_item_id])
        if (user_id, neg_item_id) in user_item_features_dict:
            neg_features.update(user_item_features_dict[(user_id, neg_item_id)])
        if neg_item_id in category_features_dict:
            neg_features.update(category_features_dict[neg_item_id])

        neg_features["target"] = 0
        neg_features["user_id"] = user_id
        neg_features["item_id"] = neg_item_id
        train_examples.append(neg_features)

    # Создаем DataFrame
    train_df = pd.DataFrame(train_examples)

    # Заполняем пропуски только для числовых
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    train_df[numeric_cols] = train_df[numeric_cols].fillna(0)

    log_message(
        f"Данные подготовлены. Размер: {len(train_df)} (положительных: {len(test_sample)}, негативных: {len(test_sample)})"
    )
    return train_df


def load_and_process_embeddings(
    items_ddf, embedding_column="fclip_embed", device="cuda", max_items=0
):
    """
    Потоковая обработка эмбеддингов для больших таблиц.
    Возвращает словарь item_id -> np.array
    """
    log_message("Оптимизированная потоковая загрузка эмбеддингов...")

    if max_items > 0:
        items_sample = items_ddf[["item_id", embedding_column]].head(
            max_items, compute=True
        )
    else:
        items_sample = items_ddf[["item_id", embedding_column]].compute()

    embeddings_dict = {}
    for row in tqdm(
        items_sample.itertuples(index=False),
        total=len(items_sample),
        desc="Обработка эмбеддингов",
    ):
        item_id = row.item_id
        embedding_data = getattr(row, embedding_column, None)
        if embedding_data is None:
            continue
        try:
            if isinstance(embedding_data, str):
                embedding = np.fromstring(
                    embedding_data.strip("[]"), sep=",", dtype=np.float32
                )
            elif isinstance(embedding_data, list):
                embedding = np.array(embedding_data, dtype=np.float32)
            elif isinstance(embedding_data, np.ndarray):
                embedding = embedding_data.astype(np.float32)
            else:
                continue
            if embedding.size > 0:
                embeddings_dict[item_id] = embedding
        except Exception:
            continue

    log_message(f"Загружено эмбеддингов для {len(embeddings_dict)} товаров")
    return embeddings_dict


# -------------------- Основной запуск --------------------
if __name__ == "__main__":
    start_time = time.time()

    # Создаем файл для логирования
    log_file = "/home/root6/python/e_cup/rec_system/training_log.txt"

    def log_message(message):
        """Функция для логирования сообщений в файл и вывод в консоль"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(full_message + "\n")

    # Очищаем файл лога при каждом запуске
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(
            f"=== НАЧАЛО ОБУЧЕНИЯ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n"
        )

    try:
        K = 100
        RECENT_N = 5
        TEST_SIZE = 0.2

        # Параметры масштабирования
        SCALING_STAGE = "full"  # small, medium, large, full

        scaling_config = {
            "small": {"sample_users": 500, "sample_fraction": 0.1},
            "medium": {"sample_users": 5000, "sample_fraction": 0.3},
            "large": {"sample_users": 20000, "sample_fraction": 0.7},
            "full": {"sample_users": None, "sample_fraction": 1.0},
        }

        config = scaling_config[SCALING_STAGE]

        log_message(f"=== РЕЖИМ МАСШТАБИРОВАНИЯ: {SCALING_STAGE.upper()} ===")
        log_message(f"Пользователей: {config['sample_users'] or 'все'}")
        log_message(f"Данных: {config['sample_fraction']*100}%")

        # === ЗАГРУЗКА ДАННЫХ ===
        stage_start = time.time()
        log_message("=== ЗАГРУЗКА ДАННЫХ ===")
        orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_ddf = (
            load_train_data()
        )
        orders_ddf, tracker_ddf, items_ddf = filter_data(
            orders_ddf, tracker_ddf, items_ddf
        )
        stage_time = time.time() - stage_start
        log_message(f"Загрузка данных завершена за {timedelta(seconds=stage_time)}")

        # === ЗАГРУЗКА ЭМБЕДДИНГОВ ===
        stage_start = time.time()
        log_message("=== ЗАГРУЗКА ЭМБЕДДИНГОВ ===")
        embeddings_dict = load_and_process_embeddings(items_ddf)
        stage_time = time.time() - stage_start
        log_message(
            f"Загрузка эмбеддингов завершена за {timedelta(seconds=stage_time)}"
        )
        log_message(f"Загружено эмбеддингов: {len(embeddings_dict)}")

        # === SPLIT ДАННЫХ ===
        stage_start = time.time()
        log_message("=== SPLIT ДАННЫХ ===")
        orders_df_full = orders_ddf.compute()
        train_orders_df, test_orders_df, cutoff_ts_per_user = train_test_split_by_time(
            orders_df_full, TEST_SIZE
        )
        stage_time = time.time() - stage_start
        log_message(f"Split данных завершен за {timedelta(seconds=stage_time)}")
        log_message(
            f"Train orders: {len(train_orders_df)}, Test orders: {len(test_orders_df)}"
        )

        # === ПОДГОТОВКА ВЗАИМОДЕЙСТВИЙ ===
        stage_start = time.time()
        log_message("=== ПОДГОТОВКА ВЗАИМОДЕЙСТВИЙ ===")
        interactions_files = prepare_interactions(
            train_orders_df, tracker_ddf, cutoff_ts_per_user, scale_days=30
        )
        stage_time = time.time() - stage_start
        log_message(
            f"Подготовка взаимодействий завершена за {timedelta(seconds=stage_time)}"
        )
        log_message(f"Создано файлов взаимодействий: {len(interactions_files)}")

        # === ПОСЛЕДНИЕ ТОВАРЫ ===
        stage_start = time.time()
        log_message("=== ПОСЛЕДНИЕ ТОВАРЫ ===")
        batch_dir = "/home/root6/python/e_cup/rec_system/data/processed/prepare_interactions_batches"
        recent_items_map = build_recent_items_map_from_batches(
            batch_dir, recent_n=RECENT_N
        )
        stage_time = time.time() - stage_start
        log_message(
            f"Построение recent items map завершено за {timedelta(seconds=stage_time)}"
        )
        log_message(f"Пользователей с recent items: {len(recent_items_map)}")

        # === ОБУЧЕНИЕ ALS ДЛЯ ПРИЗНАКОВ ===
        stage_start = time.time()
        log_message("=== ОБУЧЕНИЕ ALS ДЛЯ ПРИЗНАКОВ ===")
        model, user_map, item_map = train_als(
            interactions_files, n_factors=64, reg=1e-3, device="cuda"
        )
        inv_item_map = {v: k for k, v in item_map.items()}
        popularity_s = compute_global_popularity(
            orders_df_full, cutoff_ts_per_user
        )  # теперь это pd.Timestamp
        popular_items = popularity_s.index.tolist()
        stage_time = time.time() - stage_start
        log_message(f"Обучение ALS завершено за {timedelta(seconds=stage_time)}")
        log_message(f"Пользователей: {len(user_map)}, Товаров: {len(item_map)}")

        # === ПОСТРОЕНИЕ ДОПОЛНИТЕЛЬНЫХ ДАННЫХ ===
        stage_start = time.time()
        log_message("=== ПОСТРОЕНИЕ ДОПОЛНИТЕЛЬНЫХ ДАННЫХ ===")

        # Строим co-purchase map
        copurchase_map = build_copurchase_map(train_orders_df)
        log_message(f"Co-purchase map построен: {len(copurchase_map)} товаров")

        # Строим категорийные маппинги
        items_df = items_ddf.compute()
        categories_df = categories_ddf.compute()
        item_to_cat, cat_to_items = build_category_maps(items_df, categories_df)
        log_message(
            f"Категорийные маппинги построены: {len(item_to_cat)} товаров, {len(cat_to_items)} категорий"
        )

        stage_time = time.time() - stage_start
        log_message(
            f"Построение дополнительных данных завершено за {timedelta(seconds=stage_time)}"
        )

        # === ПРЕДВАРИТЕЛЬНЫЙ РАСЧЕТ ПРИЗНАКОВ ДЛЯ LGBM ===
        stage_start = time.time()
        log_message("=== ПРЕДВАРИТЕЛЬНЫЙ РАСЧЕТ ПРИЗНАКОВ ДЛЯ LGBM ===")

        # User features
        user_start = time.time()
        user_features_dict = build_user_features_dict(interactions_files, orders_ddf)
        user_time = time.time() - user_start
        log_message(
            f"User features построены за {timedelta(seconds=user_time)}: {len(user_features_dict)} пользователей"
        )

        # Item features
        item_start = time.time()
        item_features_dict = build_item_features_dict(
            interactions_files, items_df, orders_ddf, embeddings_dict
        )
        item_time = time.time() - item_start
        log_message(
            f"Item features построены за {timedelta(seconds=item_time)}: {len(item_features_dict)} товаров"
        )

        # User-Item features - распределенный подход
        ui_start = time.time()
        ui_features_dir = build_user_item_features_dict(
            interactions_files,
            output_dir="/home/root6/python/e_cup/rec_system/data/processed/ui_features_distributed",
        )

        if ui_features_dir is None:
            log_message("⚠️ User-Item features не созданы, пропускаем этап.")
        else:
            # Проверяем что директория создана и есть файлы
            metadata_path = os.path.join(ui_features_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                file_count = len(metadata.get("ui_feature_files", []))
                log_message(f"User-Item features созданы в: {ui_features_dir}")
                log_message(f"Количество файлов: {file_count}")
            else:
                log_message("⚠️ Метаданные UI-признаков не найдены")
                ui_features_dir = None

        ui_time = time.time() - ui_start
        log_message(f"User-Item features построены за {timedelta(seconds=ui_time)}")

        stage_time = time.time() - stage_start
        log_message(
            f"Предварительный расчет признаков завершен за {timedelta(seconds=stage_time)}"
        )

        # === ПОДГОТОВКА ДАННЫХ ДЛЯ LightGBM ===
        stage_start = time.time()
        log_message("=== ПОДГОТОВКА ДАННЫХ ДЛЯ LightGBM ===")
        recommender = LightGBMRecommender()
        recommender.set_als_embeddings(model)
        recommender.set_additional_data(
            copurchase_map, item_to_cat, cat_to_items, user_map, item_map
        )

        if embeddings_dict:
            recommender.set_external_embeddings(embeddings_dict)

        # МАСШТАБИРУЕМ данные
        if config["sample_users"]:
            sample_test_orders = test_orders_df.sample(
                min(config["sample_users"], len(test_orders_df)), random_state=42
            )
        else:
            sample_test_orders = test_orders_df

        # Используем обновленный метод с UI-признаками
        train_data, val_data = recommender.prepare_training_data(
            interactions_files=interactions_files,
            orders_ddf=orders_ddf,
            user_map=user_map,
            item_map=item_map,
            popularity_s=popularity_s,
            recent_items_map=recent_items_map,
            sample_fraction=config["sample_fraction"],
            negatives_per_positive=3,
            ui_features_dir=ui_features_dir,
            val_days=7,
        )

        # Разделяем на train/validation
        users = train_data["user_id"].unique()
        train_users, val_users = train_test_split(users, test_size=0.2, random_state=42)

        train_df = train_data[train_data["user_id"].isin(train_users)]
        val_df = train_data[train_data["user_id"].isin(val_users)]

        log_message(f"Размер train: {len(train_df)}, validation: {len(val_df)}")
        log_message(f"Признаки: {len(recommender.feature_columns)}")
        stage_time = time.time() - stage_start
        log_message(
            f"Подготовка данных для LightGBM завершена за {timedelta(seconds=stage_time)}"
        )

        # === ОБУЧЕНИЕ LightGBM ===
        stage_start = time.time()
        log_message("=== ОБУЧЕНИЕ LightGBM ===")
        model = recommender.train(train_df, val_df)
        stage_time = time.time() - stage_start
        log_message(f"Обучение LightGBM завершено за {timedelta(seconds=stage_time)}")

        # === ОЦЕНКА МОДЕЛИ ===
        stage_start = time.time()
        log_message("=== ОЦЕНКА МОДЕЛИ ===")
        train_ndcg = recommender.evaluate(train_df)
        val_ndcg = recommender.evaluate(val_df)

        log_message(f"NDCG@100 train: {train_ndcg:.4f}")
        log_message(f"NDCG@100 val: {val_ndcg:.4f}")
        stage_time = time.time() - stage_start
        log_message(f"Оценка модели завершена за {timedelta(seconds=stage_time)}")

        # Анализ важности признаков
        stage_start = time.time()
        log_message("=== ВАЖНОСТЬ ПРИЗНАКОВ ===")
        feature_importance = pd.DataFrame(
            {
                "feature": recommender.feature_columns,
                "importance": recommender.model.feature_importance(),
            }
        )
        feature_importance = feature_importance.sort_values(
            "importance", ascending=False
        )
        top_features = feature_importance.head(20)
        log_message("Топ-20 важных признаков:")
        for i, row in top_features.iterrows():
            log_message(f"  {row['feature']}: {row['importance']}")
        stage_time = time.time() - stage_start
        log_message(
            f"Анализ важности признаков завершен за {timedelta(seconds=stage_time)}"
        )

        # === СОХРАНЕНИЕ МОДЕЛИ И ВАЖНЫХ ДАННЫХ ===
        stage_start = time.time()
        log_message("=== СОХРАНЕНИЕ МОДЕЛИ И ПРИЗНАКОВ ===")
        save_data = {
            "lgbm_model": recommender.model,
            "feature_columns": recommender.feature_columns,
            "als_model": model,
            "user_map": user_map,
            "item_map": item_map,
            "inv_item_map": inv_item_map,
            "popular_items": popular_items,
            "user_features_dict": user_features_dict,
            "item_features_dict": item_features_dict,
            "ui_features_dir": ui_features_dir,  # сохраняем путь к UI-признакам
            "recent_items_map": recent_items_map,
            "copurchase_map": copurchase_map,
            "item_to_cat": item_to_cat,
        }

        model_path = (
            "/home/root6/python/e_cup/rec_system/src/models/lgbm_model_full.pkl"
        )
        with open(model_path, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        stage_time = time.time() - stage_start
        log_message(f"Сохранение модели завершено за {timedelta(seconds=stage_time)}")
        log_message(f"Модель сохранена в: {model_path}")

        # === ФИНАЛЬНАЯ СТАТИСТИКА ===
        total_time = time.time() - start_time
        log_message("=== ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО ===")
        log_message(f"Общее время выполнения: {timedelta(seconds=total_time)}")
        log_message(f"Пользователей: {len(user_map)}")
        log_message(f"Товаров: {len(item_map)}")
        log_message(f"Признаков: {len(recommender.feature_columns)}")
        log_message(f"NDCG@100 train: {train_ndcg:.4f}")
        log_message(f"NDCG@100 val: {val_ndcg:.4f}")

        # Информация о системе
        log_message("=== СИСТЕМНАЯ ИНФОРМАЦИЯ ===")
        try:
            # Информация о GPU
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                log_message(f"Видеокарта: {gpu_name}")
                log_message(f"Память GPU: {gpu_memory:.1f} GB")
            else:
                log_message("Видеокарта: CUDA не доступна")
        except Exception:
            log_message("Видеокарта: информация недоступна")

        try:
            # Информация о CPU
            import multiprocessing

            import psutil

            cpu_freq = psutil.cpu_freq()
            cpu_cores = multiprocessing.cpu_count()
            log_message(f"Процессор: {psutil.cpu_percent()}% загрузки")
            log_message(f"Ядра CPU: {cpu_cores}")
            if cpu_freq:
                log_message(f"Частота CPU: {cpu_freq.current:.1f} MHz")
        except Exception:
            log_message("Процессор: информация недоступна")

        try:
            # Информация о RAM
            import psutil

            ram = psutil.virtual_memory()
            ram_total = ram.total / 1024**3
            ram_used = ram.used / 1024**3
            log_message(
                f"Оперативная память: {ram_total:.1f} GB всего, {ram_used:.1f} GB использовано"
            )
            log_message(f"Частота RAM: информация требует дополнительных библиотек")
        except Exception:
            log_message("Оперативная память: информация недоступна")

        # Финальное сообщение
        log_message("==========================================")
        log_message("ВСЕ ЭТАПЫ ВЫПОЛНЕНЫ УСПЕШНО!")
        log_message("==========================================")

    except Exception as e:
        error_time = time.time() - start_time
        log_message(f"!!! ОШИБКА ВЫПОЛНЕНИЯ !!!")
        log_message(f"Ошибка: {str(e)}")
        log_message(f"Время до ошибки: {timedelta(seconds=error_time)}")
        log_message("Трассировка ошибки:")
        import traceback

        traceback_str = traceback.format_exc()
        log_message(traceback_str)

    finally:
        # Всегда записываем итоговое время
        total_time = time.time() - start_time
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(
                f"\n=== ОБЩЕЕ ВРЕМЯ ВЫПОЛНЕНИЯ: {timedelta(seconds=total_time)} ===\n"
            )

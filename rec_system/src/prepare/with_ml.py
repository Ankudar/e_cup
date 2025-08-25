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
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import dask.dataframe as dd
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import psutil
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

# 1000 строк ок
# 1_000_000 строк ок скор 0.0002
# 100_000_000 строк ок скор 0.0005
# 1_000_000_000 строк ок скор


# -------------------- Загрузка данных --------------------
def load_train_data(max_parts=0, max_rows=1_000_000_000):
    """
    Загружаем parquet-файлы orders, tracker, items, categories_tree, test_users.
    Ищем рекурсивно по папкам все .parquet файлы. При max_rows берём первые строки
    из нескольких партиций, а не только из первой.
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

    def find_parquet_files(folder):
        files = glob(os.path.join(folder, "**", "*.parquet"), recursive=True)
        files.sort()  # стабильный порядок "с начала"
        return files

    def read_sample(
        folder, columns=None, name="", max_parts=max_parts, max_rows=max_rows
    ):
        files = find_parquet_files(folder)
        if not files:
            print(f"{name}: parquet файлы не найдены в {folder}")
            return dd.from_pandas(pd.DataFrame(), npartitions=1)

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
        current_dtypes = dtype_profiles.get(name, {})

        ddf = dd.read_parquet(
            files,
            engine="pyarrow",
            dtype=current_dtypes if current_dtypes else None,
            gather_statistics=False,
            split_row_groups=True,
        )

        if columns is not None:
            available_cols = [c for c in columns if c in ddf.columns]
            if not available_cols:
                print(f"{name}: ни одна из колонок {columns} не найдена, пропускаем")
                return dd.from_pandas(pd.DataFrame(), npartitions=1)
            ddf = ddf[available_cols]

        total_parts = ddf.npartitions

        # --- режим "все строки" - оставляем как есть
        if max_rows == 0:
            out_ddf = ddf
            used_parts = total_parts
            count = out_ddf.shape[0].compute()  # вычисляем количество строк
        else:
            # --- НОВЫЙ ПОДХОД: используем Dask для выборки без загрузки в память ---
            if max_parts == 0:
                # Берем все партиции, но ограничиваем строки
                out_ddf = ddf.head(max_rows, compute=False)  # НЕ вычисляем сразу!
                used_parts = total_parts
            else:
                # Ограничиваем и партиции и строки
                parts_to_read = min(max_parts, total_parts)
                # Создаем новый Dask DataFrame из первых N партиций
                out_ddf = ddf.partitions[:parts_to_read].head(max_rows, compute=False)
                used_parts = parts_to_read

            # Вычисляем приблизительное количество строк (без загрузки в память)
            count = out_ddf.shape[0].compute()

        # Вычисляем использование памяти приблизительно
        mem_estimate = out_ddf.memory_usage(deep=True).sum().compute() / (1024**2)

        print(
            f"{name}: {count:,} строк (использовано {used_parts} из {total_parts} партиций), ~{mem_estimate:.1f} MB"
        )
        return out_ddf

    print("Загружаем данные...")
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
    print("Данные загружены")

    return orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_ddf


# -------------------- Фильтрация данных --------------------
def filter_data(orders_ddf, tracker_ddf, items_ddf):
    """
    Фильтруем: берём только доставленные заказы и действия page_view, favorite, to_cart.
    """
    print("Фильтрация данных...")
    orders_ddf = orders_ddf[orders_ddf["last_status"] == "delivered_orders"]
    tracker_ddf = tracker_ddf[
        tracker_ddf["action_type"].isin(["page_view", "favorite", "to_cart"])
    ]
    print("Фильтрация завершена")
    return orders_ddf, tracker_ddf, items_ddf


# -------------------- Train/Test split по времени --------------------
def train_test_split_by_time(orders_df, test_size=0.2):
    """
    Temporal split на train/test для каждого пользователя без Python loop.
    """
    print("Делаем train/test split...")
    orders_df = orders_df.copy()
    orders_df["created_timestamp"] = pd.to_datetime(orders_df["created_timestamp"])
    orders_df = orders_df.sort_values(["user_id", "created_timestamp"])

    user_counts = orders_df.groupby("user_id")["item_id"].transform("count")
    test_counts = (user_counts * test_size).astype(int).clip(lower=1)

    orders_df["cumcount"] = orders_df.groupby("user_id").cumcount()
    max_cum = orders_df.groupby("user_id")["cumcount"].transform("max")

    mask_test = orders_df["cumcount"] >= (max_cum + 1 - test_counts)

    train_df = orders_df.loc[~mask_test].drop(columns="cumcount")
    test_df = orders_df.loc[mask_test].drop(columns="cumcount")

    cutoff_ts_per_user = test_df.groupby("user_id")["created_timestamp"].min().to_dict()

    print("Split завершён")
    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        cutoff_ts_per_user,
    )


# -------------------- Подготовка взаимодействий --------------------
def prepare_interactions(
    train_orders_df,
    tracker_ddf,
    cutoff_ts_per_user,
    batch_size=300_000_000,
    action_weights=None,
    scale_days=60,
    output_dir="/home/root6/python/e_cup/rec_system/data/processed/prepare_interactions_batches",
):
    print("Формируем матрицу взаимодействий по батчам...")

    if action_weights is None:
        action_weights = {"page_view": 2, "favorite": 4, "to_cart": 6}

    os.makedirs(output_dir, exist_ok=True)
    batch_files = []
    ref_time = train_orders_df["created_timestamp"].max()

    # ====== Orders ======
    print("... для orders")
    n_rows = len(train_orders_df)
    for start in range(0, n_rows, batch_size):
        batch = train_orders_df.iloc[start : start + batch_size].copy()
        days_ago = (ref_time - batch["created_timestamp"]).dt.days.clip(lower=1)
        time_factor = np.log1p(days_ago / scale_days)
        batch = batch.assign(
            timestamp=batch["created_timestamp"],
            weight=5.0 * time_factor,
        )[["user_id", "item_id", "weight", "timestamp"]]

        path = os.path.join(output_dir, f"orders_batch_{start}.parquet")
        batch.to_parquet(path, index=False, engine="pyarrow")
        batch_files.append(path)
        del batch
        gc.collect()
        print(f"Сохранен orders-батч {start}-{min(start+batch_size, n_rows)}")

    # ====== Tracker ====== # ИСПРАВЛЕНИЕ ЗДЕСЬ
    print("... для tracker")
    tracker_ddf = tracker_ddf[["user_id", "item_id", "timestamp", "action_type"]]

    # ИСПРАВЛЕННЫЙ БЛОК: Итерируемся по партициям Dask DataFrame, а не по строкам.
    n_partitions = tracker_ddf.npartitions
    for partition_id in range(n_partitions):
        # Вычисляем одну партицию
        part = tracker_ddf.get_partition(partition_id).compute()
        part["timestamp"] = pd.to_datetime(part["timestamp"])
        part["cutoff"] = part["user_id"].map(cutoff_ts_per_user)

        mask = part["cutoff"].isna() | (part["timestamp"] < part["cutoff"])
        part = part.loc[mask]

        if part.empty:
            continue

        aw = part["action_type"].map(action_weights).fillna(0)
        days_ago = (ref_time - part["timestamp"]).dt.days.clip(lower=1)
        time_factor = np.log1p(days_ago / scale_days)
        part = part.assign(weight=aw * time_factor)[
            ["user_id", "item_id", "weight", "timestamp"]
        ]

        path = os.path.join(output_dir, f"tracker_part_{partition_id}.parquet")
        part.to_parquet(path, index=False, engine="pyarrow")
        batch_files.append(path)
        del part
        gc.collect()
        print(f"Сохранен tracker-партиция {partition_id}")

    print("Все батчи сохранены на диск.")
    return batch_files


# -------------------- Глобальная популярность --------------------
def compute_global_popularity(orders_df, cutoff_ts_per_user):
    """
    Считает популярность товаров на основе ТОЛЬКО тренировочных заказов.

    Args:
        orders_df: Все заказы (до split)
        cutoff_ts_per_user: Словарь с cutoff-временем для каждого пользователя
    """
    print("Считаем глобальную популярность на основе тренировочных данных...")

    # Фильтруем заказы: оставляем только те, что ДО cutoff времени для каждого пользователя
    train_orders = []
    for user_id, cutoff_ts in cutoff_ts_per_user.items():
        user_orders = orders_df[
            (orders_df["user_id"] == user_id)
            & (orders_df["created_timestamp"] < cutoff_ts)
        ]
        train_orders.append(user_orders)

    train_orders_df = pd.concat(train_orders, ignore_index=True)

    # Считаем популярность только на тренировочных данных
    pop = (
        train_orders_df.groupby("item_id")["item_id"]
        .count()
        .sort_values(ascending=False)
    )
    popularity = pop / pop.max()
    print(
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
    print("Первый проход: построение маппингов...")

    for f in tqdm(interactions_files):
        df = pl.read_parquet(f, columns=["user_id", "item_id"])
        user_set.update(df["user_id"].unique().to_list())
        item_set.update(df["item_id"].unique().to_list())

    user_map = {u: i for i, u in enumerate(sorted(user_set))}
    item_map = {i: j for j, i in enumerate(sorted(item_set))}
    print(f"Маппинги построены. Уников: users={len(user_map)}, items={len(item_map)}")

    # 2. ПРОХОД: Сохранение батчей на указанный диск
    print("Сохранение батчей на диск...")

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
    print("Постепенное обучение...")

    als_model = TorchALS(len(user_map), len(item_map), n_factors=64, device="cuda")

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
            print(f"Ошибка обработки батча {batch_path}: {e}")
            continue

    # Опционально: удаляем временные файлы после обучения
    print("Очистка временных файлов...")
    for batch_path in batch_files:
        try:
            os.remove(batch_path)
        except Exception as e:
            print(f"Ошибка удаления файла {batch_path}: {e}")

    # Проверяем, пуста ли директория и удаляем ее
    try:
        if not os.listdir(batch_dir):
            os.rmdir(batch_dir)
            print("Директория батчей удалена")
        else:
            print("В директории остались файлы, не удаляем")
    except Exception as e:
        print(f"Ошибка удаления директории: {e}")

    print("Обучение завершено!")
    return als_model, user_map, item_map


def build_copurchase_map(
    train_orders_df, min_co_items=2, top_n=20, device="cuda", max_items=1_000_000
):
    """
    строим словарь совместных покупок для топ-N товаров
    """
    print("Строим co-purchase матрицу для топ-N товаров...")

    # 1. Находим топ-10000 популярных товаров
    item_popularity = train_orders_df["item_id"].value_counts()
    top_items = item_popularity.head(max_items).index.tolist()
    popular_items_set = set(top_items)

    print(f"Топ-{len(top_items)} популярных товаров определены")

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
        print("Нет корзин с популярными товарами")
        return {}

    print(f"Обрабатываем {len(baskets)} корзин с популярными товарами")

    # 3. Словарь {item_id -> index} только для популярных товаров
    item2idx = {it: i for i, it in enumerate(top_items)}
    idx2item = {i: it for it, i in item2idx.items()}
    n_items = len(top_items)

    print(f"Уникальных популярных товаров: {n_items}")

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
        print("Нет данных для построения матрицы")
        return {}

    print(f"Создаем sparse матрицу из {len(rows)} взаимодействий...")

    rows_tensor = torch.tensor(rows, dtype=torch.long, device=device)
    cols_tensor = torch.tensor(cols, dtype=torch.long, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)

    co_matrix = torch.sparse_coo_tensor(
        torch.stack([rows_tensor, cols_tensor]),
        values_tensor,
        size=(n_items, n_items),
        device=device,
    ).coalesce()  # объединяем дубликаты

    print(
        f"Sparse матрица построена: {co_matrix.shape}, ненулевых элементов: {co_matrix._nnz()}"
    )

    # 6. Нормализация построчно
    row_sums = torch.sparse.sum(co_matrix, dim=1).to_dense().clamp(min=1e-9)

    # 7. Формируем финальный словарь топ-N для каждого item
    final_copurchase = {}
    indices = co_matrix.indices()
    values = co_matrix.values()

    print("Формируем рекомендации...")
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

    print(f"Co-purchase словарь построен для {len(final_copurchase)} товаров")

    # 8. Статистика
    avg_recommendations = sum(len(v) for v in final_copurchase.values()) / max(
        1, len(final_copurchase)
    )
    print(f"В среднем {avg_recommendations:.1f} рекомендаций на товар")

    return final_copurchase


def build_category_maps(items_df, categories_df):
    """
    Ускоренная версия: строим маппинги товаров и категорий.
    """
    print("Построение категорийных маппингов...")

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
            print(f"Ошибка обработки файла {f}: {e}")
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

    print(f"✅ Модель сохранена: {path}")


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
                print(f"Partial fit epoch {epoch}, Loss: {total_loss.item():.6f}")


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
            print(f"Ошибка загрузки UI-признаков для пар: {e}")
            return None

    def _add_rich_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Добавление расширенных фич для обучения модели."""

        print("Добавление богатых признаков...")

        # === ВРЕМЕННЫЕ ФИЧИ ===
        if "timestamp" in data.columns:
            try:
                data["is_weekend"] = data["timestamp"].dt.dayofweek >= 5
                data["hour"] = data["timestamp"].dt.hour
            except Exception as e:
                print(f"Не удалось преобразовать timestamp: {e}")
                data["is_weekend"] = 0
                data["hour"] = -1
        else:
            print("⚠️ Внимание: в данных нет колонки 'timestamp'. Фичи будут NaN.")
            data["is_weekend"] = np.nan
            data["hour"] = np.nan

        # === ПОПУЛЯРНОСТЬ ТОВАРА ===
        item_pop = (
            data.groupby("item_id")["user_id"]
            .count()
            .rename("item_popularity")
            .reset_index()
        )
        data = data.merge(item_pop, on="item_id", how="left")

        # === ПОПУЛЯРНОСТЬ КАТЕГОРИИ ===
        if "category_id" in data.columns:
            cat_pop = (
                data.groupby("category_id")["user_id"]
                .count()
                .rename("category_popularity")
                .reset_index()
            )
            data = data.merge(cat_pop, on="category_id", how="left")
        else:
            data["category_popularity"] = 0

        # === КОВИЗИТЫ ===
        if (
            hasattr(self, "covisitation_matrix")
            and self.covisitation_matrix is not None
        ):
            data["covisitation_score"] = data["item_id"].map(
                self.covisitation_matrix.get, na_action="ignore"
            )
            data["covisitation_score"] = data["covisitation_score"].fillna(0)
        else:
            data["covisitation_score"] = 0

        # === FCLIP ЭМБЕДДИНГИ С ИСПОЛЬЗОВАНИЕМ GPU ===
        if self.external_embeddings_dict:
            print("Ускоренная обработка FCLIP эмбеддингов на GPU...")

            # Переносим эмбеддинги на GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Собираем item_ids и размерность
            all_item_ids = list(self.external_embeddings_dict.keys())
            sample_embedding = next(iter(self.external_embeddings_dict.values()))
            embedding_dim = len(sample_embedding)
            n_fclip_dims = min(10, embedding_dim)

            # Тензор [n_items, embedding_dim] на GPU
            embeddings_tensor = torch.zeros(
                len(all_item_ids), embedding_dim, device=device
            )
            for idx, item_id in enumerate(all_item_ids):
                embeddings_tensor[idx] = torch.tensor(
                    self.external_embeddings_dict[item_id],
                    device=device,
                    dtype=torch.float32,
                )

            # Маппинг item_id → index
            item_id_to_idx = {item_id: idx for idx, item_id in enumerate(all_item_ids)}

            # Обработка батчами
            batch_size = 100000
            total_rows = len(data)

            for i in range(n_fclip_dims):
                print(f"Обработка FCLIP измерения {i+1}/{n_fclip_dims} на GPU...")

                # Создаём колонку заранее
                data[f"fclip_embed_{i}"] = 0.0

                # Обрабатываем батчами
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch_data = data.iloc[start_idx:end_idx]

                    # item_ids батча
                    batch_item_ids = batch_data["item_id"].values

                    # Маска для товаров, у которых есть эмбеддинги
                    valid_mask = np.array(
                        [item_id in item_id_to_idx for item_id in batch_item_ids]
                    )
                    valid_indices = np.where(valid_mask)[0]
                    valid_item_ids = batch_item_ids[valid_mask]

                    if len(valid_item_ids) > 0:
                        # Индексы в тензоре
                        tensor_indices = [
                            item_id_to_idx[item_id] for item_id in valid_item_ids
                        ]
                        tensor_indices = torch.tensor(tensor_indices, device=device)

                        # Извлекаем измерение эмбеддингов
                        batch_embeddings = (
                            embeddings_tensor[tensor_indices, i].cpu().numpy()
                        )

                        # Заполняем значения
                        data.iloc[
                            start_idx + valid_indices,
                            data.columns.get_loc(f"fclip_embed_{i}"),
                        ] = batch_embeddings

                    # Очистка
                    del batch_data, batch_item_ids
                    if start_idx % (batch_size * 5) == 0:
                        torch.cuda.empty_cache()

            # Освобождаем память
            del embeddings_tensor, item_id_to_idx
            torch.cuda.empty_cache()

        # === РЕГИСТРАЦИЯ НОВЫХ ПРИЗНАКОВ В feature_columns ===
        new_features = [
            "is_weekend",
            "hour",
            "item_popularity",
            "category_popularity",
            "covisitation_score",
        ] + [f"fclip_embed_{i}" for i in range(10)]

        # Добавляем только те фичи, которые есть в данных и которых еще нет в feature_columns
        existing_features = set(self.feature_columns)
        for feature in new_features:
            if feature in data.columns and feature not in existing_features:
                self.feature_columns.append(feature)
                existing_features.add(feature)

        print(f"Добавлены фичи: {[f for f in new_features if f in data.columns]}")
        print(f"Всего фич после добавления: {len(self.feature_columns)}")

        return data

    def prepare_training_data(
        self,
        interactions_files,
        user_map,
        item_map,
        popularity_s,
        recent_items_map,
        test_orders_df,
        sample_fraction=0.1,
        negatives_per_positive=1,
        ui_features_dir=None,
    ):
        print("Подготовка данных для LightGBM...")
        test_orders_df = test_orders_df.sample(frac=sample_fraction, random_state=42)

        # 1. Загрузка взаимодействий
        print("Быстрая загрузка взаимодействий...")
        interactions_chunks = [
            pd.read_parquet(f, columns=["user_id", "item_id", "timestamp", "weight"])
            for f in tqdm(interactions_files, desc="Загрузка взаимодействий")
        ]
        interactions_df = pd.concat(interactions_chunks, ignore_index=True)
        interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
        max_timestamp = interactions_df["timestamp"].max()

        # 2. Позитивные примеры
        positive_df = test_orders_df[["user_id", "item_id"]].copy()
        positive_df["target"] = 1

        # 3. Добавляем timestamp к позитивным примерам
        positive_with_time = positive_df.merge(
            interactions_df[["user_id", "item_id", "timestamp"]].drop_duplicates(
                subset=["user_id", "item_id"]
            ),
            on=["user_id", "item_id"],
            how="left",
        )

        # Заполняем пропущенные timestamp максимальным значением
        positive_with_time["timestamp"] = positive_with_time["timestamp"].fillna(
            max_timestamp
        )

        # 4. История взаимодействий
        print("Сбор истории взаимодействий...")
        user_interacted_items = (
            interactions_df.groupby("user_id")["item_id"].agg(set).to_dict()
        )
        user_positive_items = (
            positive_df.groupby("user_id")["item_id"].agg(set).to_dict()
        )

        # 5. Подготовка множеств для сэмплинга
        all_items = np.array(list(item_map.keys()))
        popular_items = set(popularity_s.nlargest(10000).index.tolist())

        print("Векторизованное негативное сэмплирование...")

        negative_samples = []

        for user_id, pos_items in tqdm(
            user_positive_items.items(), desc="Негативные сэмплы"
        ):
            interacted = user_interacted_items.get(user_id, set())
            excluded = pos_items | interacted

            available_mask = ~np.isin(all_items, list(excluded))
            available_items = all_items[available_mask]

            if len(available_items) == 0:
                continue

            n_neg = negatives_per_positive
            popular_mask = np.isin(available_items, list(popular_items))
            popular_candidates = available_items[popular_mask]
            random_candidates = available_items[~popular_mask]

            sampled_items = []

            if len(popular_candidates) > 0:
                sampled_items.extend(
                    np.random.choice(
                        popular_candidates,
                        min(n_neg // 2, len(popular_candidates)),
                        replace=False,
                    )
                )
            if len(random_candidates) > 0:
                sampled_items.extend(
                    np.random.choice(
                        random_candidates,
                        min(n_neg - len(sampled_items), len(random_candidates)),
                        replace=False,
                    )
                )

            negative_samples.extend(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": max_timestamp,  # timestamp для негативных примеров
                    "target": 0,
                }
                for item_id in sampled_items
            )

        negative_df = pd.DataFrame(negative_samples)

        # 6. Объединяем позитивные и негативные примеры
        train_data = pd.concat([positive_with_time, negative_df], ignore_index=True)

        # 7. Перемешивание
        train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

        # 8. Базовые фичи
        train_data = self._add_rich_features(train_data)

        # 9. Добавляем UI-фичи
        if ui_features_dir and os.path.exists(ui_features_dir):
            print("Добавление распределенных UI-признаков...")
            try:
                ui_features_batch = get_ui_features_batch(
                    train_data[["user_id", "item_id"]].to_dict("records"),
                    ui_features_dir,
                )

                if ui_features_batch:
                    ui_features_df = pd.DataFrame(ui_features_batch)

                    # убираем возможные дубли колонок
                    cols_to_drop = [
                        c
                        for c in ui_features_df.columns
                        if c in train_data.columns and c not in ["user_id", "item_id"]
                    ]
                    if cols_to_drop:
                        print(f"Удаляем дубликаты UI-фич при merge: {cols_to_drop}")
                        ui_features_df = ui_features_df.drop(columns=cols_to_drop)

                    # merge
                    train_data = train_data.merge(
                        ui_features_df, on=["user_id", "item_id"], how="left"
                    ).fillna(0)

                    # регистрируем новые фичи
                    ui_feature_columns = [
                        col
                        for col in ui_features_df.columns
                        if col not in ["user_id", "item_id"]
                    ]
                    existing_cols = set(self.feature_columns)
                    new_cols = [
                        col for col in ui_feature_columns if col not in existing_cols
                    ]
                    self.feature_columns.extend(new_cols)

                    print(f"Добавлены UI признаки: {new_cols}")

            except Exception as e:
                print(f"Ошибка загрузки UI-признаков: {e}")

        print(f"Данные подготовлены: {len(train_data)} примеров")
        return train_data

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
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_at": [100],
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "num_leaves": 127,  # Увеличили для сложных признаков
                "max_depth": 8,  # Увеличили глубину
                "min_child_samples": 30,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "verbosity": 1,
                "random_state": 42,
            }

        # Группы для lambdarank (количество товаров на пользователя)
        train_groups = train_data.groupby("user_id").size().values
        X_train = train_data[self.feature_columns]
        y_train = train_data["target"]

        print(f"Размер тренировочных данных: {len(X_train)}")
        print(f"Количество групп: {len(train_groups)}")

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

        print("Начинаем обучение LightGBM...")
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
            print("Модель не обучена")
            return 0.0

        if len(data) == 0:
            print("Нет данных для оценки")
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

    print("Построение словаря пользовательских признаков...")

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
    print("Агрегация по заказам...")
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

    print(
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

    print("Построение словаря товарных признаков...")

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
    print("Добавление данных из items_df...")
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
    print("Добавление эмбеддингов...")
    for item_id, embedding in embeddings_dict.items():
        if item_id in item_stats_dict:
            for i in range(min(5, len(embedding))):
                item_stats_dict[item_id][f"fclip_embed_{i}"] = float(embedding[i])

    print(f"Словарь товарных признаков построен. Записей: {len(item_stats_dict)}")
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
        print(f"Ошибка поиска UI-признаков: {e}")
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
        print(f"Ошибка получения батча UI-признаков: {e}")
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
    print("Создание UI-признаков (данные остаются на диске)")

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
                print(f"Ошибка обработки файла {input_file}: {e}")
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

        print(f"✅ Создано {len(ui_feature_files)} файлов UI-признаков в: {output_dir}")
        return output_dir

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()
        return None


def build_category_features_dict(category_df, items_df):
    """
    Оптимизированная версия с использованием Polars
    """
    import polars as pl

    print("Построение категорийных признаков...")

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

    print(f"Категорийные признаки построены. Записей: {len(category_features_dict)}")
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
    print("Подготовка данных для обучения LightGBM...")

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

    print(
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
    print("Оптимизированная потоковая загрузка эмбеддингов...")

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

    print(f"Загружено эмбеддингов для {len(embeddings_dict)} товаров")
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
        popularity_s = compute_global_popularity(orders_df_full, cutoff_ts_per_user)
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
        train_data = recommender.prepare_training_data(
            interactions_files,
            user_map,
            item_map,
            popularity_s,
            recent_items_map,
            sample_test_orders,
            sample_fraction=config["sample_fraction"],
            ui_features_dir=ui_features_dir,  # передаем путь к UI-признакам
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

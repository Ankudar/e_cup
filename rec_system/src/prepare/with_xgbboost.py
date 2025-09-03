import gc
import glob
import glob as glob_module
import json
import math
import os
import pickle
import random
import shutil
import tempfile
import time
import traceback
import uuid
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import dask
import dask.dataframe as dd
import joblib
import numpy as np
import pandas as pd
import polars as pl
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torch.sparse
import xgboost as xgb
from dask.diagnostics import ProgressBar
from implicit.als import AlternatingLeastSquares
from polars import LazyFrame, concat
from scipy.sparse import coo_matrix, csr_matrix, vstack
from sklearn.decomposition import PCA
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*Downcasting object dtype arrays.*"
)
# tqdm интеграция с pandas
tqdm.pandas()

MAX_FILES = 0  # сколько файлов берем в работу. 0 - все
MAX_ROWS = 0  # сколько строк для каждой группы берем в работу. 0 - все
EMB_LENGHT = 150  # сколько частей от исходного эмбединга брать

# обучение
ITER_N = 2_000  # число эпох для обучения
EARLY_STOP = 50  # ранняя остановка обучения
VERBOSE_N = 10  # как часто выводить сведения об обучении
CHUNK_SIZE = 200_000  # размер чанка для инкрементального обучения


def find_parquet_files(folder):
    files = glob(os.path.join(folder, "**", "*.parquet"), recursive=True)
    files.sort()
    return files


# -------------------- Загрузка данных --------------------
def load_train_data(max_parts=MAX_FILES, max_rows=MAX_ROWS):
    """
    Загружаем parquet-файлы orders, tracker, items, categories_tree, test_users.
    Ищем рекурсивно по папкам все .parquet файлы. Ограничиваем общее количество строк.
    """

    paths = {
        "orders": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train_final_apparel_orders_data/",
        "tracker": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train_final_apparel_tracker_data/",
        "items": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train_final_apparel_items_data/",
        "categories": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train_final_categories_tree/",
        "test_users": "/home/root6/python/e_cup/rec_system/data/raw/test_users/",
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

    def read_sample(
        folder, columns=None, name="", max_parts=max_parts, max_rows=max_rows
    ):
        """
        Загружаем parquet-файлы с эффективным ограничением размера.
        """
        files = find_parquet_files(folder)
        if not files:
            log_message(f"{name}: parquet файлы не найдены в {folder}")
            return dd.from_pandas(pd.DataFrame(), npartitions=1)

        current_dtypes = dtype_profiles.get(name, {})

        # Ограничиваем количество файлов для чтения
        original_file_count = len(files)
        if max_parts > 0 and max_parts < original_file_count:
            files = files[:max_parts]
            used_files = max_parts
        else:
            used_files = original_file_count

        log_message(
            f"{name}: найдено {original_file_count} файлов, используем {used_files}"
        )

        try:
            # Читаем только необходимые колонки
            ddf = dd.read_parquet(
                files,
                engine="pyarrow",
                dtype=current_dtypes,
                columns=columns,
                gather_statistics=False,
                split_row_groups=False,
            )
        except Exception as e:
            log_message(f"{name}: ошибка при чтении parquet ({e}), пропускаем")
            return dd.from_pandas(pd.DataFrame(), npartitions=1)

        # Если не нужно ограничивать строки
        if max_rows == 0:
            try:
                count = ddf.shape[0].compute()
                mem_estimate = ddf.memory_usage(deep=True).sum().compute() / (1024**2)
                log_message(
                    f"{name}: {count:,} строк (использовано {used_files} файлов), ~{mem_estimate:.1f} MB"
                )
            except Exception as e:
                log_message(f"{name}: не удалось вычислить размер: {e}")
            return ddf

        # Если нужно ограничить строки
        try:
            # Вычисляем фактическое количество строк
            actual_rows = ddf.shape[0].compute()

            if actual_rows <= max_rows:
                # Если данных меньше лимита - используем всё
                mem_estimate = ddf.memory_usage(deep=True).sum().compute() / (1024**2)
                log_message(
                    f"{name}: {actual_rows:,} строк (использовано {used_files} файлов), ~{mem_estimate:.1f} MB"
                )
                return ddf
            else:
                # Если данных больше лимита - семплируем
                fraction = max_rows / actual_rows
                sampled_ddf = ddf.sample(frac=fraction, random_state=42)

                sampled_count = sampled_ddf.shape[0].compute()
                mem_estimate = sampled_ddf.memory_usage(deep=True).sum().compute() / (
                    1024**2
                )

                log_message(
                    f"{name}: ограничено {sampled_count:,} из {actual_rows:,} строк "
                    f"(семплирование {fraction:.1%}, использовано {used_files} файлов), ~{mem_estimate:.1f} MB"
                )
                return sampled_ddf

        except Exception as e:
            log_message(
                f"{name}: ошибка при ограничении строк ({e}), используем первые {max_rows} строк"
            )

            # Fallback: берем первые max_rows строк
            try:
                limited_ddf = ddf.head(max_rows, npartitions=-1)
                limited_count = limited_ddf.shape[0].compute()
                mem_estimate = limited_ddf.memory_usage(deep=True).sum().compute() / (
                    1024**2
                )

                log_message(
                    f"{name}: {limited_count:,} строк (взято первых {max_rows}, использовано {used_files} файлов), ~{mem_estimate:.1f} MB"
                )
                return limited_ddf
            except Exception as e2:
                log_message(
                    f"{name}: критическая ошибка, возвращаем пустой DataFrame: {e2}"
                )
                return dd.from_pandas(pd.DataFrame(), npartitions=1)

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

    return (orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_ddf)


# -------------------- Фильтрация данных --------------------
def filter_data(orders_ddf, tracker_ddf, items_ddf):
    """
    Фильтруем: оставляем delivered_orders (позитив) и canceled_orders (негатив),
    а также действия page_view, favorite, to_cart.
    Сначала удаляем дубликаты, потом применяем преобразования.
    """
    log_message("Фильтрация данных...")

    orders_ddf = (
        orders_ddf.drop_duplicates(subset=["user_id", "item_id", "last_status"])
        .loc[lambda df: df["last_status"].isin(["delivered_orders", "canceled_orders"])]
        .assign(
            target=lambda df: df["last_status"].map(
                {"delivered_orders": 1, "canceled_orders": 0}, meta=("target", "int8")
            )
        )
    )

    tracker_ddf = tracker_ddf.drop_duplicates(
        subset=["user_id", "item_id", "action_type", "timestamp"]
    ).loc[lambda df: df["action_type"].isin(["page_view", "favorite", "to_cart"])]

    items_ddf = items_ddf.drop_duplicates(subset=["item_id"])

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
    batch_size=1_000_000,  # Уменьшен размер батча
    action_weights=None,
    scale_days=5,
    output_dir="/home/root6/python/e_cup/rec_system/data/processed/prepare_interactions_batches",
    force_recreate=False,
    max_tracker_files=10,  # Ограничение файлов tracker
):
    log_message("Формируем матрицу взаимодействий по батчам...")

    if action_weights is None:
        action_weights = {"page_view": 1, "favorite": 5, "to_cart": 10}

    os.makedirs(output_dir, exist_ok=True)
    batch_files = []
    ref_time = train_orders_df["created_timestamp"].max()

    # Проверяем, есть ли уже файлы
    existing_files = glob_module.glob(os.path.join(output_dir, "*.parquet"))
    if existing_files and not force_recreate:
        log_message(
            f"Найдено {len(existing_files)} существующих файлов, пропускаем создание"
        )
        return existing_files

    # ====== Orders ======
    log_message("... обработка orders")
    n_rows = len(train_orders_df)
    orders_files = []

    for start in range(0, n_rows, batch_size):
        batch_path = os.path.join(output_dir, f"orders_batch_{start}.parquet")
        orders_files.append(batch_path)

        if os.path.exists(batch_path) and not force_recreate:
            log_message(f"✅ Orders батч {start} уже существует, пропускаем")
            batch_files.append(batch_path)
            continue

        end_idx = min(start + batch_size, n_rows)
        batch = train_orders_df.iloc[start:end_idx].copy()

        # вычисление временных факторов
        days_ago = (ref_time - batch["created_timestamp"]).dt.days.clip(lower=1)
        time_factor = np.log1p(days_ago / scale_days)

        result_batch = pd.DataFrame(
            {
                "user_id": batch["user_id"],
                "item_id": batch["item_id"],
                "weight": 5.0 * time_factor,
                "timestamp": batch["created_timestamp"],
                "action_type": "order",
            }
        )

        result_batch.to_parquet(batch_path, index=False, engine="pyarrow")
        batch_files.append(batch_path)

        log_message(f"Сохранен orders-батч {start}-{end_idx}")

        # Очистка памяти
        del batch, result_batch
        gc.collect()

    # ====== Tracker ======
    log_message("... обработка tracker")
    tracker_files = []

    # Получаем список файлов tracker и ограничиваем их количество
    tracker_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train_final_apparel_tracker_data/"
    all_tracker_files = find_parquet_files(tracker_path)

    if max_tracker_files > 0:
        all_tracker_files = all_tracker_files[:max_tracker_files]

    log_message(f"Обрабатываем {len(all_tracker_files)} файлов tracker")

    for file_idx, file_path in enumerate(all_tracker_files):
        batch_path = os.path.join(output_dir, f"tracker_file_{file_idx}.parquet")
        tracker_files.append(batch_path)

        if os.path.exists(batch_path) and not force_recreate:
            log_message(f"Tracker файл {file_idx} уже обработан, пропускаем")
            batch_files.append(batch_path)
            continue

        try:
            part = pd.read_parquet(
                file_path, columns=["user_id", "item_id", "timestamp", "action_type"]
            )

            # Фильтрация по действиям
            part = part[part["action_type"].isin(action_weights.keys())]

            if part.empty:
                # Создаем пустой файл
                pd.DataFrame(
                    columns=["user_id", "item_id", "weight", "timestamp", "action_type"]
                ).to_parquet(batch_path, index=False)
                batch_files.append(batch_path)
                continue

            # Преобразование времени
            part["timestamp"] = pd.to_datetime(part["timestamp"])
            mask = part["timestamp"] < cutoff_ts_per_user
            part = part.loc[mask]

            if part.empty:
                pd.DataFrame(
                    columns=["user_id", "item_id", "weight", "timestamp", "action_type"]
                ).to_parquet(batch_path, index=False)
                batch_files.append(batch_path)
                continue

            # Вычисление весов
            aw = part["action_type"].map(action_weights)
            days_ago = (ref_time - part["timestamp"]).dt.days.clip(lower=1)
            time_factor = np.log1p(days_ago / scale_days)

            result_part = pd.DataFrame(
                {
                    "user_id": part["user_id"],
                    "item_id": part["item_id"],
                    "weight": aw * time_factor,
                    "timestamp": part["timestamp"],
                    "action_type": part["action_type"],
                }
            )

            result_part.to_parquet(batch_path, index=False, engine="pyarrow")
            batch_files.append(batch_path)

            log_message(f"Обработан tracker файл {file_idx}: {len(part)} строк")

            # Очистка памяти
            del part, result_part
            gc.collect()

        except Exception as e:
            log_message(f"Ошибка обработки tracker файла {file_path}: {e}")
            continue

    # Проверяем целостность
    expected_files = orders_files + tracker_files
    missing_files = [f for f in expected_files if not os.path.exists(f)]

    if missing_files:
        log_message(f"Отсутствуют {len(missing_files)} файлов")
    else:
        log_message("Все файлы успешно созданы")

    log_message(f"Всего файлов взаимодействий: {len(batch_files)}")
    return batch_files


# -------------------- Глобальная популярность --------------------
def compute_global_popularity(orders_df, cutoff_ts_info):
    """
    Считает популярность товаров на основе тренировочных заказов.

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
    Версия с сохранением батчей на диск + сохранение item_map.pkl
    """
    # 1. ПРОХОД: Построение маппингов
    user_set = set()
    item_set = set()
    log_message("Первый проход: построение маппингов...")

    for f in tqdm(interactions_files):
        df = pl.scan_parquet(f).select(["user_id", "item_id"]).collect()
        user_set.update(df["user_id"].unique().to_list())
        item_set.update(df["item_id"].unique().to_list())

    user_map = {u: i for i, u in enumerate(sorted(user_set))}
    item_map = {i: j for j, i in enumerate(sorted(item_set))}
    log_message(
        f"Маппинги построены. Уников: users={len(user_map)}, items={len(item_map)}"
    )

    # Сохраняем item_map.pkl
    map_dir = "/home/root6/python/e_cup/rec_system/data/processed/"
    os.makedirs(map_dir, exist_ok=True)
    item_map_path = os.path.join(map_dir, "item_map.pkl")
    with open(item_map_path, "wb") as f:
        pickle.dump(item_map, f)
    log_message(f"item_map сохранен: {item_map_path}")

    # 2. ПРОХОД: Сохранение батчей на диск
    log_message("Сохранение батчей на диск...")

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
            batch_path = os.path.join(batch_dir, f"batch_{i:04d}.npz")
            np.savez(
                batch_path,
                rows=df["user_idx"].to_numpy().astype(np.int32),
                cols=df["item_idx"].to_numpy().astype(np.int32),
                vals=df["weight"].to_numpy().astype(np.float32),
            )
            batch_files.append(batch_path)

    # 3. Постепенная загрузка и обучение
    log_message("Обучение als_model...")

    als_model = TorchALS(
        len(user_map), len(item_map), n_factors=n_factors, device=device
    )

    for batch_path in tqdm(batch_files):
        try:
            data = np.load(batch_path)
            rows, cols, vals = data["rows"], data["cols"], data["vals"]

            indices_np = np.empty((2, len(rows)), dtype=np.int32)
            indices_np[0] = rows
            indices_np[1] = cols
            indices = torch.tensor(indices_np, dtype=torch.long, device=device)
            values = torch.tensor(vals, dtype=torch.float32, device=device)

            sparse_batch = torch.sparse_coo_tensor(
                indices, values, size=(len(user_map), len(item_map)), device=device
            )

            als_model.partial_fit(sparse_batch, iterations=5, lr=0.005)

            del sparse_batch, indices, values
            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            log_message(f"Ошибка обработки батча {batch_path}: {e}")
            continue

    # Очистка временных файлов
    log_message("Очистка временных файлов...")
    for batch_path in batch_files:
        try:
            os.remove(batch_path)
        except Exception as e:
            log_message(f"Ошибка удаления файла {batch_path}: {e}")

    try:
        if not os.listdir(batch_dir):
            os.rmdir(batch_dir)
            log_message("Директория батчей удалена")
        else:
            log_message("В директории остались файлы, не удаляем")
    except Exception as e:
        log_message(f"Ошибка удаления директории: {e}")

    log_message("Обучение завершено!")

    # Сохраняем обученную модель
    model_dir = "/home/root6/python/e_cup/rec_system/src/models/"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "als_model.pt")

    torch.save(als_model.state_dict(), model_path)
    log_message(f"als_model сохранен: {model_path}")
    return als_model, user_map, item_map


def build_copurchase_map(
    train_orders_df, min_co_items=2, top_n=30, device="cuda", max_items=2000
):
    """
    строим словарь совместных покупок для топ-N товаров
    и сохраняем его в JSON
    """
    log_message("Строим co-purchase матрицу для топ-N товаров...")

    # 1. Топ популярных товаров
    item_popularity = train_orders_df["item_id"].value_counts()
    top_items = item_popularity.head(max_items).index.tolist()
    popular_items_set = set(top_items)

    log_message(f"Топ-{len(top_items)} популярных товаров определены")

    # 2. Группировка корзин
    baskets = []
    for items in train_orders_df.groupby(["user_id", "created_timestamp"])[
        "item_id"
    ].apply(list):
        filtered_items = [item for item in items if item in popular_items_set]
        if len(filtered_items) >= min_co_items:
            baskets.append(filtered_items)

    if not baskets:
        log_message("Нет корзин с популярными товарами")
        return {}

    log_message(f"Обрабатываем {len(baskets)} корзин с популярными товарами")

    # 3. Словари индексов
    item2idx = {it: i for i, it in enumerate(top_items)}
    idx2item = {i: it for it, i in item2idx.items()}
    n_items = len(top_items)

    log_message(f"Уникальных популярных товаров: {n_items}")

    # 4. Sparse матрица
    rows, cols, values = [], [], []
    for items in tqdm(baskets, desc="Обработка корзин"):
        idxs = [item2idx[it] for it in items if it in item2idx]
        if len(idxs) < 2:
            continue

        weight = 1.0 / len(idxs)
        for i in range(len(idxs)):
            for j in range(len(idxs)):
                if i != j:
                    rows.append(idxs[i])
                    cols.append(idxs[j])
                    values.append(weight)

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
    ).coalesce()

    log_message(
        f"Sparse матрица построена: {co_matrix.shape}, ненулевых элементов: {co_matrix._nnz()}"
    )

    # 6. Нормализация
    row_sums = torch.sparse.sum(co_matrix, dim=1).to_dense().clamp(min=1e-9)

    # 7. Формируем финальный словарь
    final_copurchase = {}
    indices = co_matrix.indices()
    values = co_matrix.values()

    log_message("Формируем рекомендации...")
    for i in tqdm(range(n_items), desc="Обработка товаров"):
        mask = indices[0] == i
        if mask.any():
            col_indices = indices[1][mask]
            row_values = values[mask] / row_sums[i]

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

    avg_recommendations = sum(len(v) for v in final_copurchase.values()) / max(
        1, len(final_copurchase)
    )
    log_message(f"В среднем {avg_recommendations:.1f} рекомендаций на товар")

    return final_copurchase


def build_category_maps(items_df, categories_df):
    """
    Ускоренная версия: строим маппинги товаров и категорий и сохраняем в файлы.
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


def build_recent_items_map_from_batches(
    batch_dir,
    recent_n=5,
    save_path="/home/root6/python/e_cup/rec_system/data/processed/recent_items_map.pkl",
):
    """Версия где weight влияет на порядок items.
    save_path: путь для сохранения (если None — не сохраняем).
    """
    batch_files = sorted(Path(batch_dir).glob("*.parquet"))
    recent_items_map = {}

    for f in tqdm(batch_files, desc="Обработка батчей"):
        try:
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

            # Комбинированный score
            df = df.with_columns(
                (
                    pl.col("weight") * 0.8
                    + pl.col("ts_epoch") / pl.col("ts_epoch").max() * 0.2
                ).alias("score")
            )

            df_sorted = df.sort(["user_id", "score"], descending=[False, True])

            grouped = df_sorted.group_by("user_id").agg(
                pl.col("item_id").head(recent_n).alias("items")
            )

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

    # Сохранение в файл
    if save_path is not None:
        try:
            with open(save_path, "wb") as f:
                pickle.dump(recent_items_map, f)
            log_message(f"Словарь сохранен в {save_path}")
        except Exception as e:
            log_message(f"Ошибка сохранения {save_path}: {e}")

    return recent_items_map


# -------------------- Сохранение --------------------
def save_model(
    model,
    user_map,
    item_map,
    path="/home/root6/python/e_cup/rec_system/src/models/model_als.pkl",
):
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

    log_message(f"Модель сохранена: {path}")


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


class ModelRecommender:
    def __init__(self):
        super().__init__()
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

    def load_training_data_from_parquet(self, train_dir, val_dir):
        """Загрузка train/val данных из паркет-файлов"""
        train_files = sorted(Path(train_dir).glob("*.parquet"))
        val_files = sorted(Path(val_dir).glob("*.parquet"))

        train_dfs = [pd.read_parquet(f) for f in train_files]
        val_dfs = [pd.read_parquet(f) for f in val_files]

        train_data = pd.concat(train_dfs, ignore_index=True)
        val_data = pd.concat(val_dfs, ignore_index=True)

        return train_data, val_data

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
                data["is_weekend"] = (data["timestamp"].dt.dayofweek >= 5).astype(int)
                data["hour"] = data["timestamp"].dt.hour.fillna(0)
            except Exception as e:
                log_message(f"Не удалось преобразовать timestamp: {e}")
                data["is_weekend"] = 0
                data["hour"] = 0
        else:
            data["is_weekend"] = 0
            data["hour"] = 0

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
            if not item_pop.empty:
                data = data.merge(item_pop, on="item_id", how="left")

        data["item_popularity"] = data.get("item_popularity", 0).fillna(0)

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
            data["category_popularity"] = data.get("category_popularity", 0).fillna(0)
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
        data["user_activity"] = data.get("user_activity", 0).fillna(0)

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
        data["user_avg_item_popularity"] = data.get(
            "user_avg_item_popularity", 0
        ).fillna(0)

        # --- Ковизитация ---
        if (
            hasattr(self, "covisitation_matrix")
            and self.covisitation_matrix is not None
        ):
            data["covisitation_score"] = (
                data["item_id"].map(self.covisitation_matrix.get).fillna(0)
            )
        else:
            data["covisitation_score"] = 0

        # --- FCLIP эмбеддинги ---
        if getattr(self, "external_embeddings_dict", None):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            all_item_ids = list(self.external_embeddings_dict.keys())
            embedding_dim = len(next(iter(self.external_embeddings_dict.values())))
            n_fclip_dims = min(EMB_LENGHT, embedding_dim)
            embeddings_tensor = torch.tensor(
                [self.external_embeddings_dict[i] for i in all_item_ids],
                dtype=torch.float32,
                device=device,
            )
            item_id_to_idx = {item_id: idx for idx, item_id in enumerate(all_item_ids)}
            batch_size = 100_000
            total_rows = len(data)
            for i in range(n_fclip_dims):
                data[f"fclip_embed_{i}"] = 0.0
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch_item_ids = data.iloc[start_idx:end_idx]["item_id"].values
                    valid_mask = np.array(
                        [item in item_id_to_idx for item in batch_item_ids]
                    )
                    valid_indices = np.where(valid_mask)[0]
                    valid_item_ids = batch_item_ids[valid_mask]
                    if len(valid_item_ids):
                        tensor_indices = torch.tensor(
                            [item_id_to_idx[item] for item in valid_item_ids],
                            device=device,
                        )
                        batch_emb = embeddings_tensor[tensor_indices, i].cpu().numpy()
                        data.iloc[
                            start_idx + valid_indices,
                            data.columns.get_loc(f"fclip_embed_{i}"),
                        ] = batch_emb
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

    def debug_data_info(self, data, name="data"):
        """Выводит детальную информацию о данных"""
        if data is None or data.empty:
            log_message(f"{name}: Пусто")
            return

        log_message(f"=== ДЕБАГ ИНФОРМАЦИЯ: {name} ===")
        log_message(f"Размер: {len(data)} строк")
        log_message(f"Колонки: {list(data.columns)}")

        if hasattr(self, "feature_columns"):
            missing = [f for f in self.feature_columns if f not in data.columns]
            if missing:
                log_message(f"Отсутствуют признаки: {missing}")
            else:
                log_message(f"Все признаки присутствуют")

        # Статистика по целевой переменной
        if "target" in data.columns:
            target_counts = data["target"].value_counts()
            log_message(f"Целевая переменная: {dict(target_counts)}")

        # Пример первых нескольких строк
        if len(data) > 0:
            log_message("Пример первой строки:")
            for col in data.columns:
                if col in data.columns:
                    log_message(f"  {col}: {data[col].iloc[0]}")

    def prepare_training_data(
        self,
        train_interactions_files,
        test_interactions_files,
        train_orders_df,
        test_orders_df,
        items_df,
        user_map,
        item_map,
        popularity_s,
        recent_items_map=None,
        copurchase_map=None,
        item_to_cat=None,
        cat_to_items=None,
        user_features_dict=None,
        item_features_dict=None,
        embeddings_dict=None,
        sample_fraction=0.1,
        val_split_ratio=0.2,
    ):
        log_message("Подготовка данных для модели (streaming, polars lazy, батчи)...")

        base_dir = Path("/home/root6/python/e_cup/rec_system/data/processed/")
        train_out_dir = base_dir / "train_streaming"
        val_out_dir = base_dir / "val_streaming"
        tmp_dir = base_dir / "tmp_prepare"
        for p in [train_out_dir, val_out_dir, tmp_dir]:
            p.mkdir(parents=True, exist_ok=True)

        # --- helper для валидации ---
        def validate_item_ids(df: pl.LazyFrame, df_name: str):
            schema = df.collect_schema()
            if "item_id" not in schema:
                raise ValueError(f"[{df_name}] Нет колонки 'item_id'")
            if schema["item_id"] != pl.Utf8:
                raise TypeError(
                    f"[{df_name}] item_id должен быть Utf8, а сейчас {schema['item_id']}"
                )

        # --- helper для подготовки эмбеддингов ---
        def prepare_embeddings(embeddings_dict: dict) -> tuple[pl.LazyFrame, list[str]]:
            if not embeddings_dict:
                return None, []

            # Все item_id -> str
            item_ids = [str(k) for k in embeddings_dict.keys()]

            # Матрица (N, D)
            emb_matrix = np.stack(
                [np.array(v, dtype=np.float32) for v in embeddings_dict.values()],
                axis=0,
            )
            emb_dim = emb_matrix.shape[1]

            data = {"item_id": item_ids}
            for i in range(emb_dim):
                data[f"emb_{i}"] = emb_matrix[:, i]

            schema = {"item_id": pl.Utf8}
            schema.update({f"emb_{i}": pl.Float32 for i in range(emb_dim)})

            emb_feats_lazy = pl.DataFrame(data, schema=schema).lazy()
            emb_feat_cols = [f"emb_{i}" for i in range(emb_dim)]

            validate_item_ids(emb_feats_lazy, "embeddings")

            return emb_feats_lazy, emb_feat_cols

        # --- 1. Подготовка ТОЛЬКО train заказов ---
        train_orders_pl = (
            pl.from_pandas(train_orders_df)
            .lazy()
            .with_columns(
                [
                    pl.col("item_id").cast(pl.Utf8),
                    pl.col("user_id").cast(pl.Utf8),
                    pl.col("created_timestamp").cast(pl.Datetime("us")),
                ]
            )
        )

        # --- 2. Подготовка test заказов (ОТДЕЛЬНО, не смешивать!) ---
        if test_orders_df is not None and len(test_orders_df) > 0:
            test_orders_pl = (
                pl.from_pandas(test_orders_df)
                .lazy()
                .with_columns(
                    [
                        pl.col("item_id").cast(pl.Utf8),
                        pl.col("user_id").cast(pl.Utf8),
                        pl.col("created_timestamp").cast(pl.Datetime("us")),
                    ]
                )
            )
            # Сохраняем тестовые заказы отдельно для будущего использования
            test_max_ts = (
                test_orders_pl.select(pl.max("created_timestamp")).collect().item()
            )
            log_message(f"Максимальная дата тестовых заказов: {test_max_ts}")
        else:
            test_orders_pl = None
            log_message("Тестовые заказы отсутствуют")

        # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: используем ТОЛЬКО train заказы для разделения ---
        timestamp_stats = train_orders_pl.select(
            [
                pl.max("created_timestamp").alias("max_ts"),
                pl.min("created_timestamp").alias("min_ts"),
            ]
        ).collect()

        max_ts = timestamp_stats["max_ts"][0]
        min_ts = timestamp_stats["min_ts"][0]
        split_ts = min_ts + (max_ts - min_ts) * (1 - val_split_ratio)

        log_message(
            f"Разделение ТРЕНИРОВОЧНЫХ заказов по времени: {min_ts} -> {split_ts} (train) | {split_ts} -> {max_ts} (val)"
        )

        orders_train = train_orders_pl.filter(pl.col("created_timestamp") <= split_ts)
        orders_val = train_orders_pl.filter(pl.col("created_timestamp") > split_ts)

        # --- 3. Подготовка ВЗАИМОДЕЙСТВИЙ: только из тренировочного периода ---
        # Максимальная дата ТРЕНИРОВОЧНЫХ заказов
        max_train_ts = (
            train_orders_pl.select(pl.max("created_timestamp")).collect().item()
        )
        log_message(f"Максимальная дата тренировочных заказов: {max_train_ts}")

        # Загружаем только тренировочные взаимодействия
        train_inter_lazy = pl.concat(
            [
                pl.scan_parquet(str(f)).select(
                    ["user_id", "item_id", "timestamp", "weight"]
                )
                for f in train_interactions_files
            ]
        ).with_columns(
            [
                pl.col("item_id").cast(pl.Utf8),
                pl.col("user_id").cast(pl.Utf8),
                pl.col("timestamp").cast(pl.Datetime("us")),
            ]
        )

        # ФИЛЬТРУЕМ: только взаимодействия ДО максимальной даты тренировочных заказов
        safe_interactions = train_inter_lazy.filter(pl.col("timestamp") <= max_train_ts)

        # --- 4. Валидация: проверяем, что нет взаимодействий из тестового периода ---
        if test_interactions_files:
            test_inter_lazy = pl.concat(
                [
                    pl.scan_parquet(str(f)).select(
                        ["user_id", "item_id", "timestamp", "weight"]
                    )
                    for f in test_interactions_files
                ]
            )

            # Проверяем, что тестовые взаимодействия не смешались
            test_min_ts = test_inter_lazy.select(pl.min("timestamp")).collect().item()
            if test_min_ts <= max_train_ts:
                log_message(
                    "ВНИМАНИЕ: Тестовые взаимодействия пересекаются с тренировочным периодом!"
                )
            else:
                log_message("Тестовые взаимодействия отделены от тренировочных")

        validate_item_ids(safe_interactions, "safe_interactions")

        # --- 5. Добавляем признаки ТОЛЬКО из прошлого ---
        def add_safe_features(orders_df, interactions_df):
            """Добавляет только те взаимодействия, которые были ДО заказа"""
            result = (
                orders_df.join(
                    interactions_df,
                    on=["user_id", "item_id"],
                    how="left",
                )
                .filter(
                    # Фильтруем: либо взаимодействие было ДО заказа, либо его нет вообще
                    (pl.col("timestamp") < pl.col("created_timestamp"))
                    | (pl.col("timestamp").is_null())
                )
                .with_columns(
                    [
                        pl.col("timestamp")
                        .cast(pl.Datetime("us"))
                        .fill_null(pl.col("created_timestamp") - pl.duration(days=1)),
                        pl.col("weight").cast(pl.Float32).fill_null(0.0),
                        pl.when(pl.col("last_status") == "delivered_orders")
                        .then(1)
                        .otherwise(0)
                        .alias("target"),
                    ]
                )
            )
            return result

        # Создаем финальные датафреймы
        train_merged = add_safe_features(orders_train, safe_interactions)
        val_merged = add_safe_features(orders_val, safe_interactions)

        # --- 6. Валидация на утечки ---
        def check_leakage(df, dataset_name):
            """Проверяет наличие утечек данных из будущего"""
            leakage_count = (
                df.filter(pl.col("timestamp") > pl.col("created_timestamp"))
                .select(pl.len())
                .collect()
                .item()
            )

            log_message(f"Потенциальных утечек в {dataset_name}: {leakage_count}")

            if leakage_count > 0:
                log_message("Обнаружены взаимодействия ПОСЛЕ заказов!")
                leakage_samples = (
                    df.filter(pl.col("timestamp") > pl.col("created_timestamp"))
                    .select(["user_id", "item_id", "timestamp", "created_timestamp"])
                    .limit(3)
                    .collect()
                )
                log_message(f"Примеры утечек:\n{leakage_samples}")
            else:
                log_message(f"{dataset_name}: Утечек не обнаружено")

        # Проверяем оба набора
        log_message(f"check_leakage train")
        check_leakage(train_merged, "train")

        log_message(f"check_leakage val")
        check_leakage(val_merged, "val")

        # --- 7. User признаки (только на основе тренировочных данных) ---
        log_message("Формируем User признаки")
        user_feats_lazy, user_feat_cols = None, []
        if user_features_dict:
            # СОЗДАЕМ LAZY FRAME С USER FEATURES И ДЕЛАEM JOIN
            user_feats_data = []
            for user_id, feats in user_features_dict.items():
                user_feats_data.append(
                    {
                        "user_id": str(user_id),
                        **(feats if isinstance(feats, dict) else {}),
                    }
                )

            if user_feats_data:
                user_feats_lazy = pl.LazyFrame(user_feats_data)
                # Фильтруем только тех пользователей, которые есть в train через JOIN
                user_feats_lazy = user_feats_lazy.join(
                    train_merged.select("user_id").unique(),
                    on="user_id",
                    how="semi",  # ← ТОЛЬКО фильтрация, без дублирования данных
                )
                user_feat_cols = [
                    c
                    for c in user_feats_lazy.collect_schema().names()
                    if c != "user_id"
                ]

        # --- 8. Item признаки (только на основе тренировочных данных) ---
        log_message("Формируем Item признаки")
        item_feats_lazy, item_feat_cols = None, []
        if item_features_dict:
            # СОЗДАЕМ LAZY FRAME И ФИЛЬТРУЕМ ЧЕРЕЗ JOIN
            items_data = []
            for item_id, feats in item_features_dict.items():
                if isinstance(feats, np.ndarray):
                    feat_dict = {
                        f"item_feat_{i}": float(feats[i]) for i in range(len(feats))
                    }
                elif isinstance(feats, dict):
                    feat_dict = feats.copy()
                else:
                    feat_dict = {}
                feat_dict["item_id"] = str(item_id)
                items_data.append(feat_dict)

            if items_data:
                item_feats_lazy = pl.LazyFrame(items_data).with_columns(
                    pl.col("item_id").cast(pl.Utf8)
                )
                # Фильтруем только товары из train через JOIN
                item_feats_lazy = item_feats_lazy.join(
                    train_merged.select("item_id").unique(),
                    on="item_id",
                    how="semi",  # ← быстрая фильтрация
                )
                validate_item_ids(item_feats_lazy, "item_feats")
                item_feat_cols = [
                    c
                    for c in item_feats_lazy.collect_schema().names()
                    if c != "item_id"
                ]

        # --- 9. Эмбеддинги (только для тренировочных товаров) ---
        log_message("Формируем Эмбеддинги")
        emb_feats_lazy, emb_feat_cols = None, []
        if embeddings_dict:
            # СОЗДАЕМ ВСЕ эмбеддинги, фильтруем через JOIN позже
            emb_feats_lazy, emb_feat_cols = prepare_embeddings(embeddings_dict)
            if emb_feats_lazy is not None:
                # Фильтруем только товары из train через JOIN
                emb_feats_lazy = emb_feats_lazy.join(
                    train_merged.select("item_id").unique(),
                    on="item_id",
                    how="semi",  # ← быстрая фильтрация
                )

        # --- 10. Мержим признаки ---
        log_message(f"Объединение признаков в train")
        train_with_features = train_merged
        if user_feats_lazy is not None:
            train_with_features = train_with_features.join(
                user_feats_lazy, on="user_id", how="left"
            )
        if item_feats_lazy is not None:
            train_with_features = train_with_features.join(
                item_feats_lazy, on="item_id", how="left"
            )
        if emb_feats_lazy is not None:
            train_with_features = train_with_features.join(
                emb_feats_lazy, on="item_id", how="left"
            )

        log_message(f"Объединение признаков в val")
        val_with_features = val_merged
        if user_feats_lazy is not None:
            val_with_features = val_with_features.join(
                user_feats_lazy, on="user_id", how="left"
            )
        if item_feats_lazy is not None:
            val_with_features = val_with_features.join(
                item_feats_lazy, on="item_id", how="left"
            )
        if emb_feats_lazy is not None:
            val_with_features = val_with_features.join(
                emb_feats_lazy, on="item_id", how="left"
            )

        # --- 11. Заполняем NaN ---
        log_message(f"Заполняем NaN")
        schema = train_with_features.collect_schema()
        numeric_cols = [
            c
            for c, dtype in schema.items()
            if dtype.is_numeric() and c != "target" and c not in ["user_id", "item_id"]
        ]

        train_with_features = train_with_features.with_columns(
            [pl.col(c).fill_null(0.0) for c in numeric_cols]
        )
        val_with_features = val_with_features.with_columns(
            [pl.col(c).fill_null(0.0) for c in numeric_cols]
        )

        # --- 12. Сохраняем ---
        log_message(f"Сохранение паркета для train")
        train_with_features.sink_parquet(
            str(train_out_dir / "train.parquet"), row_group_size=100_000
        )

        log_message(f"Сохранение паркета для val")
        val_with_features.sink_parquet(
            str(val_out_dir / "val.parquet"), row_group_size=100_000
        )

        # feature_columns
        self.feature_columns = user_feat_cols + item_feat_cols + emb_feat_cols

        # Собираем статистику
        train_count = train_with_features.select(pl.len()).collect().item()
        val_count = val_with_features.select(pl.len()).collect().item()

        log_message(
            f"Данные подготовлены БЕЗ УТЕЧЕК. Train: {train_count} строк, Val: {val_count} строк"
        )
        log_message(f"Признаки: {self.feature_columns}")

        return train_out_dir, val_out_dir

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

    def train(
        self, train_data_path, val_data_path=None, params=None, chunk_size=1_000_000
    ):
        """
        Инкрементальное обучение с чтением данных напрямую из parquet
        """
        MODEL_PATH = "/home/root6/python/e_cup/rec_system/src/models/model.json"

        if params is None:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "learning_rate": 0.05,
                "max_depth": 6,
                "lambda": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "random_state": 42,
                "nthread": 8,
            }

        if not hasattr(self, "feature_columns") or not self.feature_columns:
            log_message("ОШИБКА: Нет признаков для обучения!")
            return None

        train_data_str = "/home/root6/python/e_cup/rec_system/data/processed/train_streaming/train.parquet"
        val_data_str = "/home/root6/python/e_cup/rec_system/data/processed/val_streaming/val.parquet"

        # Читаем общее количество строк без загрузки в память
        try:
            total_samples = (
                pl.scan_parquet(train_data_str).select(pl.len()).collect().item()
            )
            log_message(f"Всего данных для обработки: {total_samples} строк")
        except Exception as e:
            log_message(f"ОШИБКА при чтении parquet файла: {e}")
            log_message(f"Путь: {train_data_str}")
            return None

        booster = None
        chunk_idx = 0

        # Итеративное чтение чанков через scan_parquet + slice
        try:
            num_chunks = (total_samples + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, total_samples)

                # Читаем чанк через slice
                chunk = (
                    pl.scan_parquet(train_data_str)
                    .slice(start_idx, chunk_size)
                    .collect()
                )

                log_message(f"Чанк {chunk_idx + 1}/{num_chunks} ({len(chunk)} строк)")

                success, booster = self._process_chunk(
                    chunk, params, booster, MODEL_PATH, val_data_str
                )
                if not success:
                    log_message("Ошибка при обработке чанка, прерываем обучение")
                    return None

                booster.save_model(MODEL_PATH)
                log_message(f"Модель сохранена после чанка {chunk_idx + 1}")

        except Exception as e:
            log_message(f"ОШИБКА при чтении чанков: {e}")
            import traceback

            log_message(f"Трассировка: {traceback.format_exc()}")
            return None

        self.model = booster

        if val_data_str is not None:
            try:
                val_score = self.evaluate(val_data_str, k=100)
                log_message(f"Финальный NDCG@100 на валидации: {val_score:.6f}")
            except Exception as e:
                log_message(f"ОШИБКА при оценке модели: {e}")

        return self.model

    def _process_chunk(
        self, chunk_data, params, booster, model_path, val_data_path=None
    ):
        """Обработка и обучение на одном чанке"""
        try:
            # Конвертируем Polars DataFrame в pandas для совместимости с XGBoost
            chunk_pd = chunk_data.to_pandas()
            required_columns = self.feature_columns + ["target"]

            # Проверяем наличие всех необходимых колонок
            missing_cols = set(required_columns) - set(chunk_pd.columns)
            if missing_cols:
                log_message(f"Отсутствующие колонки в чанке: {missing_cols}")
                return False, booster

            chunk_pd = chunk_pd[required_columns].copy()

            # Оптимизация типов данных
            for col in chunk_pd.columns:
                if chunk_pd[col].dtype == "float64":
                    chunk_pd[col] = chunk_pd[col].astype("float32")
                elif chunk_pd[col].dtype == "int64":
                    chunk_pd[col] = chunk_pd[col].astype("int32")

            X_chunk = chunk_pd[self.feature_columns].values
            y_chunk = chunk_pd["target"].values

            dtrain = xgb.DMatrix(
                X_chunk, label=y_chunk, feature_names=self.feature_columns
            )

            # Подготовка валидационных данных
            evals = []
            if val_data_path is not None:  # ← проверяем не None
                # Читаем валидационные данные
                val_chunk = pl.read_parquet(val_data_path).to_pandas()
                val_chunk = val_chunk[self.feature_columns + ["target"]].copy()

                dval = xgb.DMatrix(
                    val_chunk[self.feature_columns].values,
                    label=val_chunk["target"].values,
                    feature_names=self.feature_columns,
                )
                evals.append((dval, "val"))

            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=50,
                evals=evals if evals else None,
                early_stopping_rounds=10 if evals else None,
                xgb_model=booster if booster is not None else None,
                verbose_eval=20,
            )

            # Очистка памяти
            del X_chunk, y_chunk, chunk_pd, dtrain
            if val_data_path is not None:
                del dval
            gc.collect()

            return True, booster

        except Exception as e:
            log_message(f"Ошибка при обработке чанка: {e}")
            import traceback

            log_message(f"Трассировка: {traceback.format_exc()}")
            return False, booster

    def _calculate_ndcg_fast(self, data, user_ids, k=100):
        if len(data) == 0:
            return 0.0
        ndcg_scores = []
        grouped = data.groupby("user_id")
        for user_id, group in grouped:
            if len(group) <= 1:
                continue
            top_items = group.nlargest(k, "score")
            relevance = top_items["target"].values
            dcg = relevance[0]
            for i in range(1, len(relevance)):
                dcg += relevance[i] / np.log2(i + 2)
            ideal_relevance = np.sort(relevance)[::-1]
            idcg = ideal_relevance[0]
            for i in range(1, len(ideal_relevance)):
                idcg += ideal_relevance[i] / np.log2(i + 2)
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    def evaluate(self, data_path, k=100):
        """Оценка модели на данных из parquet файла"""
        if self.model is None:
            return 0.0

        # Используем прямой путь к файлу
        eval_data_path = "/home/root6/python/e_cup/rec_system/data/processed/val_streaming/val.parquet"

        try:
            # Читаем данные
            data = pl.read_parquet(eval_data_path).to_pandas()
            required_cols = self.feature_columns + ["target", "user_id"]

            # Проверяем наличие колонок
            missing_cols = set(required_cols) - set(data.columns)
            if missing_cols:
                log_message(f"Отсутствующие колонки в данных: {missing_cols}")
                return 0.0

            data = data[required_cols].copy()

            dtest = xgb.DMatrix(
                data[self.feature_columns].values, feature_names=self.feature_columns
            )
            data["score"] = self.model.predict(dtest)
            return self._calculate_ndcg_fast(data, data["user_id"], k=k)

        except Exception as e:
            log_message(f"ОШИБКА при оценке: {e}")
            return 0.0


def build_user_features_dict(interactions_files, orders_df, device="cuda"):
    """
    Оптимизированная версия с использованием Polars
    ТОЛЬКО для тренировочных данных!
    """
    log_message(
        "Построение словаря пользовательских признаков из ТРЕНИРОВОЧНЫХ данных..."
    )

    # 1. АГРЕГАЦИЯ ПО ТРЕКЕРУ (взаимодействия из тренировочного периода)
    user_stats_list = []
    for f in tqdm(interactions_files, desc="Обработка трекера (train)"):
        df = pl.read_parquet(f)
        chunk_stats = df.group_by("user_id").agg(
            [
                pl.col("weight").count().alias("count"),
                pl.col("weight").sum().alias("sum"),
                pl.col("weight").max().alias("max"),
                pl.col("weight").min().alias("min"),
            ]
        )
        user_stats_list.append(chunk_stats)

    # 2. АГРЕГАЦИЯ ПО ЗАКАЗАМ (только тренировочные заказы)
    log_message("Агрегация по ЗАКАЗАМ (train)...")
    orders_pl = pl.from_pandas(orders_df)
    order_stats = orders_pl.group_by("user_id").agg(
        [
            pl.col("item_id").count().alias("user_orders_count"),
        ]
    )

    # 3. ОБЪЕДИНЕНИЕ ДАННЫХ
    if user_stats_list:
        all_stats = pl.concat(user_stats_list)
        final_stats = all_stats.group_by("user_id").agg(
            [
                pl.col("count").sum().alias("user_count"),
                pl.col("sum").sum().alias("user_sum"),
                pl.col("max").max().alias("user_max"),
                pl.col("min").min().alias("user_min"),
            ]
        )

        if len(order_stats) > 0:
            user_stats = final_stats.join(order_stats, on="user_id", how="full")
        else:
            user_stats = final_stats
    else:
        user_stats = order_stats

    # 4. ВЫЧИСЛЕНИЕ ПРОИЗВОДНЫХ ПРИЗНАКОВ
    user_stats = user_stats.with_columns(
        [
            pl.col("user_count").fill_null(0),
            pl.col("user_sum").fill_null(0),
            pl.col("user_orders_count").fill_null(0),
            (pl.col("user_sum") / pl.col("user_count")).alias("user_mean"),
        ]
    ).fill_nan(0)

    # 5. КОНВЕРТАЦИЯ В СЛОВАРЬ
    user_stats_dict = {}
    for row in user_stats.iter_rows(named=True):
        user_stats_dict[row["user_id"]] = {
            "user_count": row["user_count"],
            "user_mean": row["user_mean"],
            "user_sum": row["user_sum"],
            "user_max": row["user_max"],
            "user_min": row["user_min"],
            "user_orders_count": row["user_orders_count"],
        }

    log_message(
        f"Словарь пользовательских признаков построен. Записей: {len(user_stats_dict)}"
    )
    return user_stats_dict


def build_item_features_dict(
    interactions_files,
    items_df,
    orders_df,  # ТОЛЬКО тренировочные заказы!
    device="cuda",
    batch_size=1_000_000,
    temp_dir="/tmp/item_features",
):
    """
    Оптимизированная версия с батчевой обработкой и сохранением на диск
    ТОЛЬКО для тренировочных данных!
    """
    log_message("Построение словаря товарных признаков из ТРЕНИРОВОЧНЫХ данных...")

    os.makedirs(temp_dir, exist_ok=True)
    temp_stats_files = []

    # 1. БАТЧЕВАЯ АГРЕГАЦИЯ ПО ВЗАИМОДЕЙСТВИЯМ (ТОЛЬКО тренировочные)
    for i, f in enumerate(
        tqdm(interactions_files, desc="Обработка взаимодействий (train)")
    ):
        try:
            df = pl.read_parquet(f)
            for start in range(0, len(df), batch_size):
                end = min(start + batch_size, len(df))
                df_batch = df[start:end]

                chunk_stats = df_batch.group_by("item_id").agg(
                    [
                        pl.col("weight").count().alias("item_count"),
                        pl.col("weight").sum().alias("item_sum"),
                        pl.col("weight").max().alias("item_max"),
                        pl.col("weight").min().alias("item_min"),
                    ]
                )

                stats_path = os.path.join(temp_dir, f"stats_{i}_{start}.parquet")
                chunk_stats.write_parquet(stats_path)
                temp_stats_files.append(stats_path)

        except Exception as e:
            log_message(f"Ошибка обработки файла {f}: {e}")
            continue

    # 2. ОБЪЕДИНЕНИЕ СТАТИСТИК С ДИСКА
    final_stats = None
    for stats_file in tqdm(temp_stats_files, desc="Объединение статистик"):
        try:
            stats_df = pl.read_parquet(stats_file)
            if final_stats is None:
                final_stats = stats_df
            else:
                final_stats = pl.concat([final_stats, stats_df])
        except Exception as e:
            log_message(f"Ошибка чтения {stats_file}: {e}")
            continue

    if final_stats is not None and not final_stats.is_empty():
        final_stats = final_stats.group_by("item_id").agg(
            [
                pl.col("item_count").sum().alias("item_count"),
                pl.col("item_sum").sum().alias("item_sum"),
                pl.col("item_max").max().alias("item_max"),
                pl.col("item_min").min().alias("item_min"),
            ]
        )
    else:
        final_stats = pl.DataFrame()

    # 3. ОБРАБОТКА ЗАКАЗОВ (ТОЛЬКО тренировочные)
    log_message("Обработка ЗАКАЗОВ (train)...")
    orders_pl = pl.from_pandas(orders_df)
    order_stats = orders_pl.group_by("item_id").agg(
        [
            pl.col("user_id").count().alias("item_orders_count"),
        ]
    )

    # 4. ОБЪЕДИНЕНИЕ ВСЕХ ДАННЫХ
    all_item_ids = set()
    if not final_stats.is_empty():
        all_item_ids.update(final_stats["item_id"].to_list())
    if not order_stats.is_empty():
        all_item_ids.update(order_stats["item_id"].to_list())

    base_df = pl.DataFrame({"item_id": list(all_item_ids)})

    if not final_stats.is_empty():
        base_df = base_df.join(final_stats, on="item_id", how="left")
    else:
        base_df = base_df.with_columns(
            [
                pl.lit(0).alias("item_count"),
                pl.lit(0.0).alias("item_sum"),
                pl.lit(0.0).alias("item_max"),
                pl.lit(0.0).alias("item_min"),
            ]
        )

    if not order_stats.is_empty():
        base_df = base_df.join(order_stats, on="item_id", how="left")
    else:
        base_df = base_df.with_columns([pl.lit(0).alias("item_orders_count")])

    # 5. ВЫЧИСЛЕНИЕ ПРИЗНАКОВ
    base_df = base_df.with_columns(
        [
            pl.col("item_count").fill_null(0),
            pl.col("item_sum").fill_null(0),
            pl.col("item_orders_count").fill_null(0),
            pl.col("item_max").fill_null(0),
            pl.col("item_min").fill_null(0),
            (pl.col("item_sum") / pl.col("item_count")).fill_nan(0).alias("item_mean"),
        ]
    )

    # 6. СОЗДАНИЕ СЛОВАРЯ
    item_stats_dict = {}
    for row in base_df.iter_rows(named=True):
        item_stats_dict[row["item_id"]] = {
            "item_count": row["item_count"],
            "item_mean": row["item_mean"],
            "item_sum": row["item_sum"],
            "item_max": row["item_max"],
            "item_min": row["item_min"],
            "item_orders_count": row["item_orders_count"],
        }

    # 7. ОЧИСТКА
    try:
        for temp_file in temp_stats_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except Exception as e:
        log_message(f"Ошибка очистки временных файлов: {e}")

    log_message(f"Словарь товарных признаков построен. Записей: {len(item_stats_dict)}")
    return item_stats_dict


def load_and_process_embeddings(
    items_ddf,
    embedding_column="fclip_embed",
    device="cuda",
    max_items=0,
    apply_pca=True,
    pca_components=50,
):
    """
    Потоковая обработка эмбеддингов для больших таблиц с опциональным PCA.
    Возвращает словарь item_id -> np.array
    """
    log_message(
        "Оптимизированная потоковая загрузка эмбеддингов..."
        + (" с PCA" if apply_pca else " без PCA")
    )

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
        except Exception as e:
            log_message(f"Ошибка обработки эмбеддинга для item_id {item_id}: {e}")
            continue

    # Применяем PCA если требуется
    if apply_pca and embeddings_dict:
        embeddings_dict, pca_model, scaler = apply_pca_to_embeddings(
            embeddings_dict, n_components=pca_components
        )

        # Сохраняем модели PCA
        save_pca_models(
            pca_model,
            scaler,
            "/home/root6/python/e_cup/rec_system/data/processed/embeddings_pca",
        )

    log_message(f"Загружено эмбеддингов для {len(embeddings_dict)} товаров")
    return embeddings_dict


def load_streaming_data(path_pattern: str) -> pd.DataFrame:
    files = sorted(glob(path_pattern))
    if not files:
        raise FileNotFoundError(f"Нет файлов по шаблону: {path_pattern}")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def normalize_item_features_dict(item_features_dict, embed_prefix="fclip_embed_"):
    """
    Приводит словарь item_features_dict к формату {item_id: np.array([...])},
    где вектор = [обычные фичи + эмбеддинги].
    Обновлено для работы с PCA-редуцированными эмбеддингами.
    """
    normalized = {}

    # Собираем все ключи по всем item
    all_keys = set()
    for feats in item_features_dict.values():
        all_keys.update(feats.keys())

    # Разделяем на обычные и эмбеддинговые
    base_keys = [k for k in all_keys if not k.startswith(embed_prefix)]
    embed_keys = sorted(
        [k for k in all_keys if k.startswith(embed_prefix)],
        key=lambda x: int(x.replace(embed_prefix, "")),
    )

    feature_order = base_keys + embed_keys

    # Формируем numpy-вектора
    for item_id, feats in item_features_dict.items():
        row = []
        for key in feature_order:
            val = feats.get(key, 0.0)
            if val is None:
                val = 0.0
            row.append(val)
        normalized[item_id] = np.array(row, dtype=np.float32)

    return normalized, feature_order


def normalize_item_feats(item_feats, max_emb_dim=30):
    """
    Приводит фичи айтема к словарю (dict).
    Поддерживает dict и np.ndarray.
    """
    norm_feats = {}

    if isinstance(item_feats, dict):
        for k, v in item_feats.items():
            if isinstance(v, (int, float, np.number)):
                norm_feats[k] = float(v)
    elif isinstance(item_feats, np.ndarray):
        for i, v in enumerate(item_feats):
            if i < max_emb_dim:
                norm_feats[f"emb_{i}"] = float(v)

    return norm_feats


def apply_pca_to_embeddings(embeddings_dict, n_components=50, random_state=42):
    """
    Применяет PCA к эмбеддингам для уменьшения размерности

    Args:
        embeddings_dict: словарь {item_id: embedding_vector}
        n_components: количество компонент PCA
        random_state: random state для воспроизводимости

    Returns:
        dict: словарь с уменьшенными эмбеддингами
        PCA: обученная модель PCA
    """
    log_message(f"Применение PCA к эмбеддингам: {n_components} компонент")

    # Собираем все эмбеддинги в матрицу
    item_ids = list(embeddings_dict.keys())
    embeddings_matrix = np.array([embeddings_dict[item_id] for item_id in item_ids])

    # Стандартизация
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_matrix)

    # Применяем PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    embeddings_reduced = pca.fit_transform(embeddings_scaled)

    # Создаем новый словарь с уменьшенными эмбеддингами
    reduced_embeddings_dict = {}
    for i, item_id in enumerate(item_ids):
        reduced_embeddings_dict[item_id] = embeddings_reduced[i]

    # Логируем информацию о PCA
    explained_variance = np.sum(pca.explained_variance_ratio_)
    log_message(f"PCA завершено: сохранено {explained_variance:.3%} дисперсии")
    log_message(f"Исходная размерность: {embeddings_matrix.shape[1]}")
    log_message(f"Новая размерность: {n_components}")

    return reduced_embeddings_dict, pca, scaler


def save_pca_models(pca_model, scaler, base_path):
    """
    Сохраняет модели PCA и StandardScaler

    Args:
        pca_model: обученная модель PCA
        scaler: обученный StandardScaler
        base_path: базовый путь для сохранения
    """
    pca_path = f"{base_path}_pca.pkl"
    scaler_path = f"{base_path}_scaler.pkl"

    joblib.dump(pca_model, pca_path)
    joblib.dump(scaler, scaler_path)

    log_message(f"Модель PCA сохранена: {pca_path}")
    log_message(f"Модель Scaler сохранена: {scaler_path}")


def analyze_embedding_variance(embeddings_dict, max_components=None):
    """
    Анализирует дисперсию эмбеддингов для выбора оптимального числа компонент PCA
    """
    log_message("Анализ дисперсии эмбеддингов...")

    # Собираем все эмбеддинги в матрицу
    embeddings_matrix = np.array(list(embeddings_dict.values()))

    if max_components is None:
        max_components = min(embeddings_matrix.shape[1], 100)

    # Анализ дисперсии
    pca = PCA()
    pca.fit(StandardScaler().fit_transform(embeddings_matrix))

    # Кумулятивная объясненная дисперсия
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Находим оптимальное число компонент
    optimal_components = np.argmax(cumulative_variance >= 0.95) + 1
    if optimal_components == 1 and cumulative_variance[0] < 0.95:
        optimal_components = max_components

    log_message(f"Оптимальное число компонент PCA: {optimal_components}")
    log_message(
        f"Объясненная дисперсия: {cumulative_variance[optimal_components-1]:.3%}"
    )

    return optimal_components, cumulative_variance


def plot_pca_variance(embeddings_dict, save_path=None):
    """
    Строит график объясненной дисперсии для PCA
    """
    import matplotlib.pyplot as plt

    embeddings_matrix = np.array(list(embeddings_dict.values()))
    pca = PCA()
    pca.fit(StandardScaler().fit_transform(embeddings_matrix))

    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        log_message(f"График PCA сохранен: {save_path}")

    plt.close()


def fit_pca_on_sample(embeddings_dict, target_dim=50, sample_size=100_000, seed=42):
    """
    Обучает PCA на случайной выборке эмбеддингов
    """
    log_message(
        f"Обучение PCA на выборке из {min(sample_size, len(embeddings_dict))} эмбеддингов..."
    )

    rng = random.Random(seed)
    all_ids = list(embeddings_dict.keys())

    # Берём случайный сэмпл
    sample_ids = rng.sample(all_ids, min(sample_size, len(all_ids)))
    X_sample = np.array([embeddings_dict[i] for i in sample_ids], dtype=np.float32)

    # Стандартизация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    # Учим PCA
    pca = PCA(n_components=target_dim, svd_solver="randomized", random_state=seed)
    pca.fit(X_scaled)

    explained = np.sum(pca.explained_variance_ratio_)
    log_message(f"PCA: сохранено {explained:.2%} дисперсии в {target_dim} компонентах")

    return pca, scaler


def transform_embeddings_batched(embeddings_dict, pca, scaler, batch_size=10_000):
    """
    Применяет PCA преобразование к эмбеддингам батчами
    """
    log_message(f"Применение PCA ко всем {len(embeddings_dict)} эмбеддингам батчами...")

    all_ids = list(embeddings_dict.keys())
    reduced_dict = {}

    for i in tqdm(range(0, len(all_ids), batch_size), desc="PCA трансформация"):
        batch_ids = all_ids[i : i + batch_size]
        X_batch = np.array([embeddings_dict[j] for j in batch_ids], dtype=np.float32)

        # Стандартизация и PCA
        X_scaled = scaler.transform(X_batch)
        X_reduced = pca.transform(X_scaled)

        for idx, item_id in enumerate(batch_ids):
            reduced_dict[item_id] = X_reduced[idx]

    log_message(f"PCA трансформация завершена")
    return reduced_dict


def apply_pca_to_embeddings_optimized(
    embeddings_dict, n_components=50, sample_size=100000, batch_size=10000
):
    """
    Оптимизированная версия применения PCA к эмбеддингам
    """
    # Обучаем PCA на выборке
    pca_model, scaler = fit_pca_on_sample(
        embeddings_dict, target_dim=n_components, sample_size=sample_size
    )

    # Применяем ко всем данным
    reduced_embeddings = transform_embeddings_batched(
        embeddings_dict, pca_model, scaler, batch_size=batch_size
    )

    return reduced_embeddings, pca_model, scaler


def apply_pca_to_embeddings_streaming(
    items_ddf, embedding_column="fclip_embed", n_components=50, sample_size=50000
):
    """
    Потоковое применение PCA без загрузки всех эмбеддингов в память
    """
    log_message("Потоковое применение PCA к эмбеддингам...")

    # Сначала получаем выборку для обучения PCA
    sample_items = items_ddf[["item_id", embedding_column]].sample(frac=0.1).compute()

    sample_embeddings = []
    sample_ids = []

    for row in tqdm(
        sample_items.itertuples(index=False), desc="Подготовка выборки PCA"
    ):
        item_id = row.item_id
        embedding_data = getattr(row, embedding_column, None)
        if embedding_data is not None:
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
                    sample_embeddings.append(embedding)
                    sample_ids.append(item_id)
            except Exception as e:
                continue

    # Обучаем PCA на выборке
    scaler = StandardScaler()
    sample_scaled = scaler.fit_transform(sample_embeddings)
    pca = PCA(n_components=n_components)
    pca.fit(sample_scaled)

    # Теперь обрабатываем все данные потоково
    embeddings_dict = {}
    batch_size = 10000

    for i in range(0, len(items_ddf), batch_size):
        batch = items_ddf[i : i + batch_size].compute()

        for row in batch.itertuples(index=False):
            item_id = row.item_id
            embedding_data = getattr(row, embedding_column, None)

            if embedding_data is not None:
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

                    # Применяем PCA
                    embedding_scaled = scaler.transform(embedding.reshape(1, -1))
                    embedding_reduced = pca.transform(embedding_scaled)[0]

                    embeddings_dict[item_id] = embedding_reduced

                except Exception as e:
                    continue

    return embeddings_dict, pca, scaler


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

        # Сначала загружаем эмбеддинги без PCA
        raw_embeddings_dict = load_and_process_embeddings(
            items_ddf,
            apply_pca=False,  # Сначала загружаем без PCA
        )

        log_message(f"Загружено сырых эмбеддингов: {len(raw_embeddings_dict)}")
        log_message(
            f"Размерность исходных эмбеддингов: {len(next(iter(raw_embeddings_dict.values())))}"
        )

        # Анализируем дисперсию для выбора оптимального числа компонент
        optimal_components, variance = analyze_embedding_variance(raw_embeddings_dict)
        plot_pca_variance(
            raw_embeddings_dict, "/home/root6/python/e_cup/rec_system/pca_variance.png"
        )

        # Используем оптимальное число компонент или EMB_LENGHT, если оно меньше
        final_components = min(optimal_components, EMB_LENGHT)
        log_message(
            f"Используем {final_components} компонент PCA (оптимально: {optimal_components}, лимит: {EMB_LENGHT})"
        )

        # Применяем PCA с выбранным числом компонент (оптимизированная версия)
        if len(items_ddf) > 10_000_000:  # Если больше 10 млн строк
            log_message("Используем потоковый PCA для больших данных...")
            embeddings_dict, pca_model, scaler = apply_pca_to_embeddings_streaming(
                items_ddf,
                embedding_column="fclip_embed",
                n_components=final_components,
                sample_size=50000,
            )
        else:
            # Для нормальных объемов используем оптимизированный вариант
            embeddings_dict, pca_model, scaler = apply_pca_to_embeddings_optimized(
                raw_embeddings_dict,
                n_components=final_components,
                sample_size=100000,
                batch_size=10000,
            )

        # Сохраняем уменьшенные эмбеддинги
        save_path = (
            "/home/root6/python/e_cup/rec_system/data/processed/embeddings_dict.pkl"
        )
        with open(save_path, "wb") as f:
            pickle.dump(embeddings_dict, f)

        # Сохраняем модели PCA и scaler
        save_pca_models(
            pca_model,
            scaler,
            "/home/root6/python/e_cup/rec_system/data/processed/embeddings_pca",
        )

        log_message(
            f"Пример ключей в embeddings_dict: {list(embeddings_dict.keys())[:5]}"
        )
        log_message(
            f"Пример значения после PCA: {next(iter(embeddings_dict.values()))[:5]}"
        )
        log_message(
            f"Размерность после PCA: {len(next(iter(embeddings_dict.values())))}"
        )

        stage_time = time.time() - stage_start
        log_message(
            f"Загрузка эмбеддингов с PCA завершена за {timedelta(seconds=stage_time)}"
        )
        log_message(f"Загружено эмбеддингов: {len(embeddings_dict)}")

        # Очищаем память от сырых эмбеддингов
        del raw_embeddings_dict
        gc.collect()

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
            train_orders_df,
            tracker_ddf,
            cutoff_ts_per_user,
            scale_days=5,
            force_recreate=False,
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
        save_path = (
            "/home/root6/python/e_cup/rec_system/data/processed/popular_items.pkl"
        )
        with open(save_path, "wb") as f:
            pickle.dump(popular_items, f)
        stage_time = time.time() - stage_start
        log_message(f"Обучение ALS завершено за {timedelta(seconds=stage_time)}")
        log_message(f"Пользователей: {len(user_map)}, Товаров: {len(item_map)}")

        # === ПОСТРОЕНИЕ ДОПОЛНИТЕЛЬНЫХ ДАННЫХ ===
        stage_start = time.time()
        log_message("=== ПОСТРОЕНИЕ ДОПОЛНИТЕЛЬНЫХ ДАННЫХ ===")

        # Строим co-purchase map
        copurchase_map = build_copurchase_map(train_orders_df)
        save_path = (
            "/home/root6/python/e_cup/rec_system/data/processed/copurchase_map.pkl"
        )
        with open(save_path, "wb") as f:
            pickle.dump(copurchase_map, f)
        log_message(f"Co-purchase map построен: {len(copurchase_map)} товаров")

        # Строим категорийные маппинги
        items_df = items_ddf.compute()
        categories_df = categories_ddf.compute()
        item_to_cat, cat_to_items = build_category_maps(items_df, categories_df)
        save_path = "/home/root6/python/e_cup/rec_system/data/processed/item_to_cat.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(item_to_cat, f)
        save_path = (
            "/home/root6/python/e_cup/rec_system/data/processed/cat_to_items.pkl"
        )
        with open(save_path, "wb") as f:
            pickle.dump(cat_to_items, f)
        log_message(
            f"Категорийные маппинги построены: {len(item_to_cat)} товаров, {len(cat_to_items)} категорий"
        )

        stage_time = time.time() - stage_start
        log_message(
            f"Построение дополнительных данных завершено за {timedelta(seconds=stage_time)}"
        )

        # === ПРЕДВАРИТЕЛЬНЫЙ РАСЧЕТ ПРИЗНАКОВ ===
        stage_start = time.time()
        log_message("=== ПРЕДВАРИТЕЛЬНЫЙ РАСЧЕТ ПРИЗНАКОВ ===")

        # User features - используем ТОЛЬКО тренировочные данные!
        user_start = time.time()
        user_features_dict = build_user_features_dict(
            interactions_files, train_orders_df
        )  # ← ИСПРАВЛЕНО
        save_path = (
            "/home/root6/python/e_cup/rec_system/data/processed/user_features_dict.pkl"
        )
        with open(save_path, "wb") as f:
            pickle.dump(user_features_dict, f)
        user_time = time.time() - user_start
        log_message(
            f"User features построены за {timedelta(seconds=user_time)}: {len(user_features_dict)} пользователей"
        )

        # Item features - используем ТОЛЬКО тренировочные данные!
        item_start = time.time()
        raw_item_features_dict = build_item_features_dict(
            interactions_files, items_df, train_orders_df  # ← ИСПРАВЛЕНО
        )

        # Преобразуем словари в np.array
        item_features_dict, feature_order = normalize_item_features_dict(
            raw_item_features_dict
        )

        # Логируем пример
        log_message(
            f"Item features: {len(item_features_dict)} товаров, размер вектора: {len(feature_order)} признаков"
        )
        log_message(
            f"Порядок признаков: {feature_order[:10]}{'...' if len(feature_order)>10 else ''}"
        )

        sample_items = list(item_features_dict.items())[:5]
        for k, v in sample_items:
            log_message(
                f"item_features_dict[{k}] type={type(v)}, shape={v.shape}, example={v[:5]}"
            )

        # Сохраняем
        save_path = (
            "/home/root6/python/e_cup/rec_system/data/processed/item_features_dict.pkl"
        )
        with open(save_path, "wb") as f:
            pickle.dump(item_features_dict, f)

        # Отдельно сохраняем порядок признаков (важно для инференса)
        save_path = (
            "/home/root6/python/e_cup/rec_system/data/processed/item_feature_order.pkl"
        )
        with open(save_path, "wb") as f:
            pickle.dump(feature_order, f)

        item_time = time.time() - item_start
        log_message(f"Item features построены за {timedelta(seconds=item_time)}")

        # === ПОДГОТОВКА ДАННЫХ ДЛЯ модели ===
        stage_start = time.time()
        log_message("=== ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ ===")
        recommender = ModelRecommender()
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
            train_interactions_files=interactions_files,
            test_interactions_files=[],  # если нет отдельного теста
            train_orders_df=train_orders_df,
            test_orders_df=pd.DataFrame(),  # пустой DF для валидирования
            items_df=items_df,
            user_map=user_map,
            item_map=item_map,
            popularity_s=popularity_s,
            recent_items_map=recent_items_map,
            copurchase_map=copurchase_map,
            item_to_cat=item_to_cat,
            cat_to_items=cat_to_items,
            user_features_dict=user_features_dict,  # обязательно передаем
            item_features_dict=item_features_dict,  # обязательно передаем
            embeddings_dict=embeddings_dict,  # вот сюда эмбеддинги
            sample_fraction=config["sample_fraction"],
        )

        # --- Функция для подсчёта строк в parquet ---
        def parquet_len(parquet_pattern: str) -> int:
            """Безопасно возвращает количество строк в parquet (0 если файлов нет)."""
            files = list(Path().glob(parquet_pattern))
            if not files:
                return 0
            return (
                pl.scan_parquet([str(f) for f in files])
                .select(pl.len())
                .collect()[0, 0]
            )

        # --- Логирование размеров train/val ---
        log_message(f"Собираем паркет для трейна")
        train_parquet_pattern = "rec_system/data/processed/train_streaming/*.parquet"
        log_message(f"Собираем паркет для вала")
        val_parquet_pattern = "rec_system/data/processed/val_streaming/*.parquet"

        train_len = parquet_len(train_parquet_pattern)
        val_len = parquet_len(val_parquet_pattern)

        log_message(f"Размер train: {train_len}, validation: {val_len}")
        log_message(f"Признаки: {len(recommender.feature_columns)}")

        stage_time = time.time() - stage_start
        log_message(
            f"Подготовка данных для модели завершена за {timedelta(seconds=stage_time)}"
        )

        # === ДЕТАЛЬНАЯ ПРОВЕРКА ПРИЗНАКОВ ===
        stage_start = time.time()
        log_message("=== ДЕТАЛЬНАЯ ПРОВЕРКА FEATURE GENERATION ===")

        # --- Проверки user features ---
        log_message("--- ПРОВЕРКА USER FEATURES ---")
        if user_features_dict:
            sample_user = next(iter(user_features_dict))
            user_feats = user_features_dict[sample_user]
            log_message(f"Пример user features для пользователя {sample_user}:")
            for feat, value in user_feats.items():
                log_message(f"  {feat}: {value}")

            users_with_features = len(user_features_dict)
            users_with_real_features = sum(
                1
                for feats in user_features_dict.values()
                if any(v != 0 for v in feats.values())
            )
            log_message(f"Пользователей с features: {users_with_features}")
            log_message(
                f"Пользователей с НЕнулевыми features: {users_with_real_features}"
            )
        else:
            log_message("user_features_dict ПУСТОЙ!")

        # --- Проверки item features ---
        log_message("--- ПРОВЕРКА ITEM FEATURES ---")
        if item_features_dict:
            sample_item = next(iter(item_features_dict))
            raw_item_feats = item_features_dict[sample_item]
            item_feats = normalize_item_feats(raw_item_feats, max_emb_dim=EMB_LENGHT)

            log_message(f"Пример item features для товара {sample_item}:")
            for feat, value in item_feats.items():
                log_message(f"  {feat}: {value}")

            items_with_features = len(item_features_dict)
            items_with_real_features = 0
            for feats in item_features_dict.values():
                norm_feats = normalize_item_feats(feats, max_emb_dim=EMB_LENGHT)
                if any(v != 0 for v in norm_feats.values()):
                    items_with_real_features += 1

            log_message(f"Товаров с features: {items_with_features}")
            log_message(f"Товаров с НЕнулевыми features: {items_with_real_features}")
        else:
            log_message("item_features_dict ПУСТОЙ!")

        # --- Проверка эмбеддингов ---
        log_message("--- ПРОВЕРКА ЭМБЕДДИНГОВ ---")
        if embeddings_dict:
            sample_item = next(iter(embeddings_dict))
            embedding = embeddings_dict[sample_item]
            log_message(
                f"Пример эмбеддинга для товара {sample_item}: shape {embedding.shape}"
            )
            log_message(f"Эмбеддингов загружено: {len(embeddings_dict)}")
            log_message(f"Пример значений: {embedding[:5]}")
        else:
            log_message("embeddings_dict ПУСТОЙ!")

        # --- Проверка co-purchase map ---
        log_message("--- ПРОВЕРКА CO-PURCHASE MAP ---")
        if copurchase_map:
            sample_item = next(iter(copurchase_map))
            co_items = copurchase_map[sample_item]
            log_message(
                f"Пример co-purchase для товара {sample_item}: {len(co_items)} товаров"
            )
            log_message(f"Co-purchase записей: {len(copurchase_map)}")
        else:
            log_message("copurchase_map ПУСТОЙ!")

        # --- Проверка категорийных маппингов ---
        log_message("--- ПРОВЕРКА КАТЕГОРИЙНЫХ МАППИНГОВ ---")
        if item_to_cat and cat_to_items:
            sample_item = next(iter(item_to_cat))
            cat_id = item_to_cat[sample_item]
            cat_items = cat_to_items.get(cat_id, [])
            log_message(f"Товар {sample_item} -> категория {cat_id}")
            log_message(f"Категория {cat_id} -> {len(cat_items)} товаров")
            log_message(f"Товаров в маппинге: {len(item_to_cat)}")
            log_message(f"Категорий в маппинге: {len(cat_to_items)}")
        else:
            log_message("Категорийные маппинги ПУСТЫЕ!")

        stage_time = time.time() - stage_start
        log_message(f"Проверка признаков завершена за {timedelta(seconds=stage_time)}")

        # === ОБУЧЕНИЕ МОДЕЛИ ===
        stage_start = time.time()
        log_message("=== ОБУЧЕНИЕ МОДЕЛИ ===")

        train_df = load_streaming_data(train_parquet_pattern)
        val_df = load_streaming_data(val_parquet_pattern)

        # Проверяем данные
        recommender.debug_data_info(train_df, "TRAIN")
        recommender.debug_data_info(val_df, "VAL")

        # Логирование признаков
        if hasattr(recommender, "feature_columns"):
            log_message(f"Feature columns: {recommender.feature_columns}")
        else:
            log_message("Feature columns не определены!")

        # Проверяем наличие данных и признаков перед обучением
        if not train_df.empty and getattr(recommender, "feature_columns", None):
            model = recommender.train(train_df, val_df, chunk_size=CHUNK_SIZE)
        else:
            log_message("Нельзя обучать: нет данных или признаков")

        # model = recommender.train(train_df, val_df)

        stage_time = time.time() - stage_start
        log_message(f"Обучение модели завершено за {timedelta(seconds=stage_time)}")

        # === ОЦЕНКА МОДЕЛИ ===
        # stage_start = time.time()
        # log_message("=== ОЦЕНКА МОДЕЛИ ===")
        # train_ndcg = recommender.evaluate(train_df)
        # val_ndcg = recommender.evaluate(val_df)

        # log_message(f"NDCG@100 train: {train_ndcg:.4f}")
        # log_message(f"NDCG@100 val: {val_ndcg:.4f}")
        # stage_time = time.time() - stage_start
        # log_message(f"Оценка модели завершена за {timedelta(seconds=stage_time)}")

        # # Анализ важности признаков
        # stage_start = time.time()
        # log_message("=== ВАЖНОСТЬ ПРИЗНАКОВ ===")
        # feature_importance = pd.DataFrame(
        #     {
        #         "feature": recommender.feature_columns,
        #         "importance": recommender.model.feature_importance(),
        #     }
        # )
        # feature_importance = feature_importance.sort_values(
        #     "importance", ascending=False
        # )
        # top_features = feature_importance.head(20)
        # log_message("Топ-20 важных признаков:")
        # for i, row in top_features.iterrows():
        #     log_message(f"  {row['feature']}: {row['importance']}")
        # stage_time = time.time() - stage_start
        # log_message(
        #     f"Анализ важности признаков завершен за {timedelta(seconds=stage_time)}"
        # )

        # === СОХРАНЕНИЕ МОДЕЛИ И ВАЖНЫХ ДАННЫХ ===
        stage_start = time.time()
        log_message("=== СОХРАНЕНИЕ МОДЕЛИ И ПРИЗНАКОВ ===")
        save_data = {
            "model": recommender.model,
            "feature_columns": recommender.feature_columns,
            "user_map": user_map,
            "item_map": item_map,
            "inv_item_map": inv_item_map,
            "popular_items": popular_items,
            "user_features_dict": user_features_dict,
            "item_features_dict": item_features_dict,
            "recent_items_map": recent_items_map,
            "copurchase_map": copurchase_map,
            "item_to_cat": item_to_cat,
        }

        model_path = "/home/root6/python/e_cup/rec_system/src/models/model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        stage_time = time.time() - stage_start
        log_message(f"Сохранение модели завершено за {timedelta(seconds=stage_time)}")
        log_message(f"Модель сохранена в: {model_path}")

        # Финальная статистика
        all_items = set()

        # === ФИНАЛЬНАЯ СТАТИСТИКА ===
        total_time = time.time() - start_time
        log_message("=== ОБУЧЕНИЕ И ПРЕДСКАЗАНИЯ ЗАВЕРШЕНЫ УСПЕШНО ===")
        log_message(f"Общее время выполнения: {timedelta(seconds=total_time)}")
        log_message(f"Пользователей: {len(user_map)}")
        log_message(f"Товаров: {len(item_map)}")
        log_message(f"Признаков: {len(recommender.feature_columns)}")
        # log_message(f"NDCG@100 train: {train_ndcg:.4f}")
        # log_message(f"NDCG@100 val: {val_ndcg:.4f}")

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

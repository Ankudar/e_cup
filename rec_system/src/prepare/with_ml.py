import gc
import os
import pickle
import random
import time
from collections import defaultdict
from datetime import timedelta
from glob import glob
from pathlib import Path

import dask.dataframe as dd
import lightgbm as lgb
import numpy as np
import pandas as pd
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
def load_train_data(max_parts=0, max_rows=1000):
    """
    Загружаем parquet-файлы orders, tracker, items, categories_tree, test_users.
    Ищем рекурсивно по папкам все .parquet файлы.
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
        # Рекурсивно ищем все .parquet
        return [f for f in glob(os.path.join(folder, "**/*.parquet"), recursive=True)]

    def read_sample(
        folder, columns=None, name="", max_parts=max_parts, max_rows=max_rows
    ):
        files = find_parquet_files(folder)
        if not files:
            print(f"{name}: parquet файлы не найдены в {folder}")
            return dd.from_pandas(pd.DataFrame(), npartitions=1)

        # читаем все файлы
        ddf = dd.read_parquet(files)

        if columns is not None:
            available_cols = [c for c in columns if c in ddf.columns]
            if not available_cols:
                print(f"{name}: ни одна из колонок {columns} не найдена, пропускаем")
                return dd.from_pandas(pd.DataFrame(), npartitions=1)
            ddf = ddf[available_cols]

        total_parts = ddf.npartitions
        if max_parts > 0:
            used_parts = min(total_parts, max_parts)
            ddf = dd.concat([ddf.get_partition(i) for i in range(used_parts)])
        else:
            used_parts = total_parts

        # Правильное ограничение max_rows:
        if max_rows > 0:
            df_all = ddf.head(
                max_rows, compute=True
            )  # читаем только первые max_rows строк
            ddf = dd.from_pandas(df_all, npartitions=1)

        count = ddf.map_partitions(len).compute().sum()
        mem_bytes = (
            ddf.map_partitions(lambda df: df.memory_usage(deep=True).sum())
            .compute()
            .sum()
        )
        mem_mb = mem_bytes / (1024**2)
        print(
            f"{name}: {count:,} строк (использовано {used_parts} из {total_parts} партиций), ~{mem_mb:.1f} MB"
        )
        return ddf

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
    batch_size=100_000_000,
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
def compute_global_popularity(train_orders_df):
    print("Считаем глобальную популярность...")
    pop = (
        train_orders_df.groupby("item_id")["item_id"]
        .count()
        .sort_values(ascending=False)
    )
    popularity = pop / pop.max()
    print("Глобальная популярность рассчитана")
    return popularity


# -------------------- Обучение ALS --------------------
def train_als(interactions_files, n_factors=64, reg=1e-3, device="cuda"):
    """
    Обучение ALS с итеративной загрузкой данных для построения маппингов.
    Фактическое обучение происходит на сжатом sparse тензоре.
    """
    # 1. ПРОХОД: Собираем все user_id и item_id для построения маппингов
    user_set = set()
    item_set = set()
    print("Первый проход: построение маппингов...")
    for f in tqdm(interactions_files):
        # Читаем только нужные колонки
        df_chunk = pd.read_parquet(f, columns=["user_id", "item_id"])
        user_set.update(df_chunk["user_id"].unique())
        item_set.update(df_chunk["item_id"].unique())

    user_map = {u: i for i, u in enumerate(sorted(user_set))}
    item_map = {i: j for j, i in enumerate(sorted(item_set))}
    print(f"Маппинги построены. Уников: users={len(user_map)}, items={len(item_map)}")

    # 2. ПРОХОД: Построение разреженного тензора COO на GPU
    rows_list, cols_list, values_list = [], [], []
    print("Второй проход: построение тензора...")
    for f in tqdm(interactions_files):
        df_chunk = pd.read_parquet(f, columns=["user_id", "item_id", "weight"])
        # Применяем маппинги
        df_chunk["user_idx"] = df_chunk["user_id"].map(user_map)
        df_chunk["item_idx"] = df_chunk["item_id"].map(item_map)
        df_chunk = df_chunk.dropna(subset=["user_idx", "item_idx"])  # Важно!

        rows_list.append(df_chunk["user_idx"].values.astype(np.int32))
        cols_list.append(df_chunk["item_idx"].values.astype(np.int32))
        values_list.append(df_chunk["weight"].values.astype(np.float32))

    # Конкатенация и перенос на GPU
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    values = np.concatenate(values_list)

    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)
    sparse_tensor = torch.sparse_coo_tensor(
        indices, values, size=(len(user_map), len(item_map))
    )
    print(f"Тензор построен на GPU. Ненулевых элементов: {sparse_tensor._nnz()}")

    # Обучаем ALS
    als_model = TorchALS(
        len(user_map), len(item_map), n_factors=n_factors, reg=reg, device=device
    )
    als_model.fit(sparse_tensor, iterations=500, lr=0.005, sample_ratio=0.5)

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


def build_recent_items_map_from_batches(batch_dir, recent_n=5, device="cuda"):
    """
    Оптимизированная версия с обработкой на GPU
    """
    batch_files = sorted(Path(batch_dir).glob("*.parquet"))
    recent_items_map = {}

    for f in tqdm(batch_files, desc="Обработка батчей"):
        try:
            # Читаем весь файл
            df = pd.read_parquet(f, columns=["user_id", "item_id", "timestamp"])

            # ПРАВИЛЬНОЕ преобразование timestamp в числовой формат
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Используем astype вместо view
            df["timestamp_numeric"] = df["timestamp"].astype("int64") // 10**9

            # Конвертируем в тензоры на GPU
            user_ids = torch.tensor(
                df["user_id"].values, device=device, dtype=torch.long
            )
            item_ids = torch.tensor(
                df["item_id"].values, device=device, dtype=torch.long
            )
            timestamps = torch.tensor(
                df["timestamp_numeric"].values, device=device, dtype=torch.long
            )

            # Сортируем на GPU - более простая и эффективная сортировка
            # Сначала сортируем массив индексов
            indices = torch.arange(len(user_ids), device=device)

            # Сортируем сначала по user_id, затем по timestamp (по убыванию)
            sort_mask = user_ids * 10**12 + (
                10**12 - timestamps
            )  # Комбинированный ключ для сортировки
            sorted_indices = torch.argsort(sort_mask)

            user_ids = user_ids[sorted_indices]
            item_ids = item_ids[sorted_indices]

            # Находим уникальных пользователей и их counts
            unique_users, counts = torch.unique_consecutive(
                user_ids, return_counts=True
            )

            # Обрабатываем каждого пользователя
            start = 0
            for i, user_id in enumerate(unique_users.cpu().numpy()):
                count = counts[i].item()
                user_items = (
                    item_ids[start : start + count][:recent_n].cpu().numpy().tolist()
                )
                start += count

                if user_id not in recent_items_map:
                    recent_items_map[user_id] = user_items
                else:
                    # Объединяем и берем последние recent_n
                    combined = (recent_items_map[user_id] + user_items)[:recent_n]
                    recent_items_map[user_id] = combined

            # Очищаем память
            del user_ids, item_ids, timestamps, sort_mask, sorted_indices
            if device == "cuda":
                torch.cuda.empty_cache()

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
        self.to(device)

    def forward(self, user, item):
        return (self.user_factors[user] * self.item_factors[item]).sum(1)

    def fit(
        self,
        sparse_tensor,
        iterations=500,
        lr=0.005,
        show_progress=True,
        sample_ratio=0.5,
        early_stop_patience=10,
    ):
        """Обучение ALS с семплированием и ранней остановкой"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.reg)
        if not sparse_tensor.is_coalesced():
            sparse_tensor = sparse_tensor.coalesce()
        users_coo, items_coo = sparse_tensor.indices()
        values = sparse_tensor.values()
        n_interactions = len(values)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(iterations):
            sample_size = int(n_interactions * sample_ratio)
            sample_indices = torch.randint(
                0, n_interactions, (sample_size,), device=self.device
            )
            optimizer.zero_grad()
            batch_users = users_coo[sample_indices]
            batch_items = items_coo[sample_indices]
            batch_values = values[sample_indices]

            pred = self.forward(batch_users, batch_items)
            loss = F.mse_loss(pred, batch_values)

            user_reg = (
                self.reg
                * self.user_factors[batch_users].pow(2).sum()
                / sample_size
                * n_interactions
            )
            item_reg = (
                self.reg
                * self.item_factors[batch_items].pow(2).sum()
                / sample_size
                * n_interactions
            )
            total_loss = loss + user_reg + item_reg

            total_loss.backward()
            optimizer.step()

            if show_progress and (epoch % 50 == 0 or epoch == iterations - 1):
                print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")

            # Ранняя остановка
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break


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
    ):
        """
        Векторизованная подготовка данных для LightGBM с быстрым негативным сэмплированием.
        """
        print("Подготовка данных для LightGBM...")

        # 1. Ограничиваем данные
        test_orders_df = test_orders_df.sample(frac=sample_fraction, random_state=42)

        # 2. Загрузка взаимодействий
        print("Быстрая загрузка взаимодействий...")
        interactions_chunks = [
            pd.read_parquet(f, columns=["user_id", "item_id", "timestamp", "weight"])
            for f in tqdm(interactions_files, desc="Загрузка взаимодействий")
        ]
        interactions_df = pd.concat(interactions_chunks, ignore_index=True)
        interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
        max_timestamp = interactions_df["timestamp"].max()

        # 3. Позитивные примеры
        positive_df = test_orders_df[["user_id", "item_id"]].copy()
        positive_df["target"] = 1

        # 4. Построение словаря пользователь → просмотренные товары
        print("Сбор истории взаимодействий...")
        user_interacted_items = (
            interactions_df.groupby("user_id")["item_id"].agg(set).to_dict()
        )
        user_positive_items = (
            positive_df.groupby("user_id")["item_id"].agg(set).to_dict()
        )

        # 5. Подготовка множеств для быстрого сэмплирования
        all_items = np.array(list(item_map.keys()))
        popular_items = set(popularity_s.nlargest(10000).index.tolist())

        print("Векторизованное негативное сэмплирование...")

        negative_samples = []

        # Генерируем негативы по пользователям
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
                {"user_id": user_id, "item_id": item_id, "target": 0}
                for item_id in sampled_items
            )

        negative_df = pd.DataFrame(negative_samples)

        # 6. Объединение с позитивными
        train_data = pd.concat([positive_df, negative_df], ignore_index=True)

        # 7. Перемешиваем
        train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

        # 8. Добавление богатых признаков
        train_data = self._add_rich_features(
            train_data,
            interactions_df,
            popularity_s,
            recent_items_map,
            user_map,
            item_map,
            max_timestamp,
        )

        print(f"Данные подготовлены: {len(train_data)} примеров")
        return train_data

    def _add_rich_features(
        self,
        data,
        interactions_df,
        popularity_s,
        recent_items_map,
        user_map,
        item_map,
        max_timestamp,
    ):
        """
        Добавление богатых признаков к данным
        """
        print("Добавление богатых признаков...")

        # Базовые признаки
        data["item_popularity"] = (
            data["item_id"].map(lambda x: popularity_s.get(x, 0.0)).fillna(0.0)
        )

        # === ПРИЗНАКИ ПОЛЬЗОВАТЕЛЯ ===
        user_stats = (
            interactions_df.groupby("user_id")
            .agg(
                {
                    "weight": ["count", "mean", "sum", "std", "max", "min"],
                    "timestamp": ["max", "min"],
                }
            )
            .reset_index()
        )
        user_stats.columns = [
            "user_id",
            "user_count",
            "user_mean",
            "user_sum",
            "user_std",
            "user_max",
            "user_min",
            "user_last_ts",
            "user_first_ts",
        ]

        # Время с последнего взаимодействия
        user_stats["user_days_since_last"] = (
            max_timestamp - user_stats["user_last_ts"]
        ).dt.days
        user_stats["user_days_since_first"] = (
            max_timestamp - user_stats["user_first_ts"]
        ).dt.days

        data = data.merge(user_stats, on="user_id", how="left")

        # === ПРИЗНАКИ ТОВАРА ===
        item_stats = (
            interactions_df.groupby("item_id")
            .agg(
                {
                    "weight": ["count", "mean", "sum", "std", "max", "min"],
                    "timestamp": ["max", "min"],
                }
            )
            .reset_index()
        )
        item_stats.columns = [
            "item_id",
            "item_count",
            "item_mean",
            "item_sum",
            "item_std",
            "item_max",
            "item_min",
            "item_last_ts",
            "item_first_ts",
        ]

        # Время с последнего взаимодействия с товаром
        item_stats["item_days_since_last"] = (
            max_timestamp - item_stats["item_last_ts"]
        ).dt.days
        item_stats["item_days_since_first"] = (
            max_timestamp - item_stats["item_first_ts"]
        ).dt.days

        data = data.merge(item_stats, on="item_id", how="left")

        # === ПРИЗНАКИ ВЗАИМОДЕЙСТВИЯ ПОЛЬЗОВАТЕЛЬ-ТОВАР ===
        user_item_stats = (
            interactions_df.groupby(["user_id", "item_id"])
            .agg(
                {
                    "weight": ["count", "mean", "sum", "std", "max", "min"],
                    "timestamp": ["max", "min"],
                }
            )
            .reset_index()
        )
        user_item_stats.columns = [
            "user_id",
            "item_id",
            "ui_count",
            "ui_mean",
            "ui_sum",
            "ui_std",
            "ui_max",
            "ui_min",
            "ui_last_ts",
            "ui_first_ts",
        ]

        user_item_stats["ui_days_since_last"] = (
            max_timestamp - user_item_stats["ui_last_ts"]
        ).dt.days
        user_item_stats["ui_days_since_first"] = (
            max_timestamp - user_item_stats["ui_first_ts"]
        ).dt.days

        data = data.merge(user_item_stats, on=["user_id", "item_id"], how="left")

        # === КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ ===
        if self.item_to_cat:
            data["item_category"] = data["item_id"].map(self.item_to_cat)
            # One-hot encoding для топ-20 категорий
            top_categories = data["item_category"].value_counts().head(20).index
            for cat in top_categories:
                data[f"cat_{cat}"] = (data["item_category"] == cat).astype(int)

        # === CO-PURCHASE ПРИЗНАКИ ===
        if self.copurchase_map:
            data["copurchase_strength"] = data.apply(
                lambda row: self._get_copurchase_strength(row["item_id"]), axis=1
            )

            # Признаки на основе истории пользователя
            data["user_copurchase_affinity"] = data.apply(
                lambda row: self._get_user_copurchase_affinity(
                    row["user_id"], row["item_id"]
                ),
                axis=1,
            )

        # === ВРЕМЕННЫЕ ПРИЗНАКИ ===
        data["is_weekend"] = data.apply(
            lambda row: (
                1
                if pd.notna(row.get("ui_last_ts")) and row["ui_last_ts"].dayofweek >= 5
                else 0
            ),
            axis=1,
        )

        data["hour_of_day"] = data.apply(
            lambda row: (
                row["ui_last_ts"].hour if pd.notna(row.get("ui_last_ts")) else 12
            ),
            axis=1,
        )

        # === RECENT ITEMS ПРИЗНАКИ ===
        data["in_recent_items"] = data.apply(
            lambda row: (
                1 if row["item_id"] in recent_items_map.get(row["user_id"], []) else 0
            ),
            axis=1,
        )

        data["recent_items_count"] = data["user_id"].map(
            lambda x: len(recent_items_map.get(x, []))
        )

        # === ALS ЭМБЕДДИНГИ ===
        if self.user_embeddings is not None and self.item_embeddings is not None:
            print("Добавление ALS эмбеддингов...")
            embedding_size = min(10, self.user_embeddings.shape[1])

            for i in range(embedding_size):
                # User embeddings
                data[f"user_als_{i}"] = data["user_id"].map(
                    lambda x: (
                        self.user_embeddings[user_map[x], i]
                        if x in user_map and user_map[x] < self.user_embeddings.shape[0]
                        else 0.0
                    )
                )

                # Item embeddings
                data[f"item_als_{i}"] = data["item_id"].map(
                    lambda x: (
                        self.item_embeddings[item_map[x], i]
                        if x in item_map and item_map[x] < self.item_embeddings.shape[0]
                        else 0.0
                    )
                )

        # === FCLIP ЭМБЕДДИНГИ С ИСПОЛЬЗОВАНИЕМ GPU ===
        if self.external_embeddings_dict:
            print("Ускоренная обработка FCLIP эмбеддингов на GPU...")

            # Переносим эмбеддинги на GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Создаем тензор всех эмбеддингов на GPU
            all_item_ids = list(self.external_embeddings_dict.keys())
            sample_embedding = next(iter(self.external_embeddings_dict.values()))
            embedding_dim = len(sample_embedding)
            n_fclip_dims = min(10, embedding_dim)

            # Создаем тензор всех эмбеддингов [n_items, embedding_dim]
            embeddings_tensor = torch.zeros(
                len(all_item_ids), embedding_dim, device=device
            )
            for idx, item_id in enumerate(all_item_ids):
                embeddings_tensor[idx] = torch.tensor(
                    self.external_embeddings_dict[item_id],
                    device=device,
                    dtype=torch.float32,
                )

            # Создаем маппинг item_id -> index в тензоре
            item_id_to_idx = {item_id: idx for idx, item_id in enumerate(all_item_ids)}

            # Обрабатываем данные батчами на GPU
            batch_size = 100000
            total_rows = len(data)

            for i in range(n_fclip_dims):
                print(f"Обработка FCLIP измерения {i+1}/{n_fclip_dims} на GPU...")

                # Создаем колонку заранее
                data[f"fclip_embed_{i}"] = 0.0

                # Обрабатываем батчами
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch_data = data.iloc[start_idx:end_idx]

                    # Получаем item_ids для батча
                    batch_item_ids = batch_data["item_id"].values

                    # Создаем маску для товаров, которые есть в эмбеддингах
                    valid_mask = np.array(
                        [item_id in item_id_to_idx for item_id in batch_item_ids]
                    )
                    valid_indices = np.where(valid_mask)[0]
                    valid_item_ids = batch_item_ids[valid_mask]

                    if len(valid_item_ids) > 0:
                        # Получаем индексы в тензоре эмбеддингов
                        tensor_indices = [
                            item_id_to_idx[item_id] for item_id in valid_item_ids
                        ]
                        tensor_indices = torch.tensor(tensor_indices, device=device)

                        # Извлекаем нужное измерение эмбеддингов
                        batch_embeddings = (
                            embeddings_tensor[tensor_indices, i].cpu().numpy()
                        )

                        # Заполняем значения
                        data.iloc[
                            start_idx + valid_indices,
                            data.columns.get_loc(f"fclip_embed_{i}"),
                        ] = batch_embeddings

                    # Очистка памяти
                    del batch_data, batch_item_ids
                    if start_idx % (batch_size * 5) == 0:
                        torch.cuda.empty_cache()

            # Освобождаем GPU память
            del embeddings_tensor, item_id_to_idx
            torch.cuda.empty_cache()

        # === ИНТЕРАКЦИОННЫЕ ПРИЗНАКИ ===
        data["user_item_affinity"] = data["ui_mean"] * data["user_mean"]
        data["popularity_affinity"] = data["item_popularity"] * data["user_count"]

        # Взаимодействия пользователя с категорией
        if self.item_to_cat:
            user_category_stats = (
                interactions_df.merge(
                    pd.DataFrame(
                        {
                            "item_id": list(self.item_to_cat.keys()),
                            "category": list(self.item_to_cat.values()),
                        }
                    ),
                    on="item_id",
                )
                .groupby(["user_id", "category"])["weight"]
                .sum()
                .reset_index()
            )

            user_category_pivot = user_category_stats.pivot_table(
                index="user_id", columns="category", values="weight", fill_value=0
            )
            user_category_pivot.columns = [
                f"user_cat_{col}" for col in user_category_pivot.columns
            ]

            data = data.merge(user_category_pivot, on="user_id", how="left")

        # Заполняем пропуски
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(0)

        # Удаляем временные колонки
        data = data.drop(
            columns=[col for col in data.columns if "_ts" in col], errors="ignore"
        )
        data = data.drop(columns=["item_category"], errors="ignore")

        self.feature_columns = [
            col
            for col in data.columns
            if col not in ["user_id", "item_id", "target", "timestamp"]
            and data[col].dtype in [np.int64, np.int32, np.float64, np.float32]
        ]

        print(f"Создано {len(self.feature_columns)} богатых признаков")
        return data

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
    Создает словарь {user_id: {feature_name: value}} с признаками пользователей.
    Использует interactions_files (трекер) и orders_df (заказы).
    """
    print("Построение словаря пользовательских признаков...")
    user_stats_dict = {}

    # 1. АГРЕГАЦИЯ ПО ТРЕКЕРУ (взаимодействия)
    for f in tqdm(interactions_files, desc="Обработка трекера"):
        df_chunk = pd.read_parquet(f)

        # Группируем по пользователю
        chunk_stats = df_chunk.groupby("user_id").agg(
            {
                "weight": ["count", "mean", "sum", "std", "max", "min"],
                "timestamp": ["max", "min"],
            }
        )

        # Обновляем общий словарь
        for user_id, stats in chunk_stats.iterrows():
            if user_id not in user_stats_dict:
                user_stats_dict[user_id] = {
                    "user_count": 0,
                    "user_mean": 0,
                    "user_sum": 0,
                    "user_std": 0,
                    "user_max": 0,
                    "user_min": 0,
                    "user_last_ts": pd.Timestamp.min,
                    "user_first_ts": pd.Timestamp.max,
                    "user_orders_count": 0,
                    "user_avg_order_value": 0,
                }

            # Обновляем статистику по взаимодействиям
            user_stats_dict[user_id]["user_count"] += stats[("weight", "count")]
            user_stats_dict[user_id]["user_sum"] += stats[("weight", "sum")]
            user_stats_dict[user_id]["user_max"] = max(
                user_stats_dict[user_id]["user_max"], stats[("weight", "max")]
            )
            user_stats_dict[user_id]["user_min"] = min(
                user_stats_dict[user_id]["user_min"], stats[("weight", "min")]
            )
            user_stats_dict[user_id]["user_last_ts"] = max(
                user_stats_dict[user_id]["user_last_ts"], stats[("timestamp", "max")]
            )
            user_stats_dict[user_id]["user_first_ts"] = min(
                user_stats_dict[user_id]["user_first_ts"], stats[("timestamp", "min")]
            )

    # 2. АГРЕГАЦИЯ ПО ЗАКАЗАМ
    print("Агрегация по заказам...")
    order_stats = (
        orders_df.groupby("user_id")
        .agg({"item_id": "count", "created_timestamp": ["min", "max"]})
        .reset_index()
    )

    order_stats.columns = [
        "user_id",
        "user_orders_count",
        "user_first_order_ts",
        "user_last_order_ts",
    ]

    for _, row in order_stats.iterrows():
        user_id = row["user_id"]
        if user_id not in user_stats_dict:
            user_stats_dict[user_id] = {
                "user_count": 0,
                "user_mean": 0,
                "user_sum": 0,
                "user_std": 0,
                "user_max": 0,
                "user_min": 0,
                "user_last_ts": pd.Timestamp.min,
                "user_first_ts": pd.Timestamp.max,
                "user_orders_count": 0,
                "user_avg_order_value": 0,
            }

        user_stats_dict[user_id]["user_orders_count"] = row["user_orders_count"]
        user_stats_dict[user_id]["user_last_order_ts"] = row["user_last_order_ts"]
        user_stats_dict[user_id]["user_first_order_ts"] = row["user_first_order_ts"]

    # 3. ВЫЧИСЛЕНИЕ ПРОИЗВОДНЫХ ПРИЗНАКОВ
    print("Вычисление производных признаков...")
    current_time = pd.Timestamp.now()

    for user_id in user_stats_dict:
        stats = user_stats_dict[user_id]

        # Время с последнего взаимодействия
        if stats["user_last_ts"] > pd.Timestamp.min:
            stats["user_days_since_last"] = (current_time - stats["user_last_ts"]).days
        else:
            stats["user_days_since_last"] = 365  # большое значение если нет данных

        # Время с первого взаимодействия
        if stats["user_first_ts"] < pd.Timestamp.max:
            stats["user_days_since_first"] = (
                current_time - stats["user_first_ts"]
            ).days
        else:
            stats["user_days_since_first"] = 365

        # Время с последнего заказа
        if "user_last_order_ts" in stats and pd.notna(stats["user_last_order_ts"]):
            stats["user_days_since_last_order"] = (
                current_time - stats["user_last_order_ts"]
            ).days
        else:
            stats["user_days_since_last_order"] = 365

        # Средний вес взаимодействия
        if stats["user_count"] > 0:
            stats["user_mean"] = stats["user_sum"] / stats["user_count"]
        else:
            stats["user_mean"] = 0

    print(
        f"Словарь пользовательских признаков построен. Записей: {len(user_stats_dict)}"
    )
    return user_stats_dict


def build_item_features_dict(
    interactions_files, items_df, orders_df, embeddings_dict, device="cuda"
):
    """
    Создает словарь {item_id: {feature_name: value}} с признаками товаров.
    """
    print("Построение словаря товарных признаков...")
    item_stats_dict = {}

    # 1. АГРЕГАЦИЯ ПО ТРЕКЕРУ И ЗАКАЗАМ (популярность)
    for f in tqdm(interactions_files, desc="Обработка взаимодействий"):
        df_chunk = pd.read_parquet(f)

        chunk_stats = df_chunk.groupby("item_id").agg(
            {
                "weight": ["count", "mean", "sum", "std", "max", "min"],
                "timestamp": ["max", "min"],
            }
        )

        for item_id, stats in chunk_stats.iterrows():
            if item_id not in item_stats_dict:
                item_stats_dict[item_id] = {
                    "item_count": 0,
                    "item_mean": 0,
                    "item_sum": 0,
                    "item_std": 0,
                    "item_max": 0,
                    "item_min": 0,
                    "item_last_ts": pd.Timestamp.min,
                    "item_first_ts": pd.Timestamp.max,
                    "item_orders_count": 0,
                }

            item_stats_dict[item_id]["item_count"] += stats[("weight", "count")]
            item_stats_dict[item_id]["item_sum"] += stats[("weight", "sum")]
            item_stats_dict[item_id]["item_max"] = max(
                item_stats_dict[item_id]["item_max"], stats[("weight", "max")]
            )
            item_stats_dict[item_id]["item_min"] = min(
                item_stats_dict[item_id]["item_min"], stats[("weight", "min")]
            )
            item_stats_dict[item_id]["item_last_ts"] = max(
                item_stats_dict[item_id]["item_last_ts"], stats[("timestamp", "max")]
            )
            item_stats_dict[item_id]["item_first_ts"] = min(
                item_stats_dict[item_id]["item_first_ts"], stats[("timestamp", "min")]
            )

    # 2. ДОБАВЛЕНИЕ ДАННЫХ ИЗ ЗАКАЗОВ
    order_item_stats = (
        orders_df.groupby("item_id").agg({"user_id": "count"}).reset_index()
    )
    order_item_stats.columns = ["item_id", "item_orders_count"]

    for _, row in order_item_stats.iterrows():
        item_id = row["item_id"]
        if item_id not in item_stats_dict:
            item_stats_dict[item_id] = {
                "item_count": 0,
                "item_mean": 0,
                "item_sum": 0,
                "item_std": 0,
                "item_max": 0,
                "item_min": 0,
                "item_last_ts": pd.Timestamp.min,
                "item_first_ts": pd.Timestamp.max,
                "item_orders_count": 0,
            }
        item_stats_dict[item_id]["item_orders_count"] = row["item_orders_count"]

    # 3. ДОБАВЛЕНИЕ ДАННЫХ ИЗ items_df
    print("Добавление данных из items_df...")
    items_features = (
        items_df.drop_duplicates(subset=["item_id"])
        .set_index("item_id")[["catalogid"]]
        .to_dict("index")
    )

    for item_id, features in items_features.items():
        if item_id not in item_stats_dict:
            item_stats_dict[item_id] = {
                "item_count": 0,
                "item_mean": 0,
                "item_sum": 0,
                "item_std": 0,
                "item_max": 0,
                "item_min": 0,
                "item_last_ts": pd.Timestamp.min,
                "item_first_ts": pd.Timestamp.max,
                "item_orders_count": 0,
                "item_category": 0,
            }
        item_stats_dict[item_id]["item_category"] = features["catalogid"]

    # 4. ДОБАВЛЕНИЕ ЭМБЕДДИНГОВ (первые N компонент)
    print("Добавление эмбеддингов...")
    for item_id, embedding in embeddings_dict.items():
        if item_id not in item_stats_dict:
            continue

        for i in range(min(5, len(embedding))):
            item_stats_dict[item_id][f"fclip_embed_{i}"] = embedding[i]

    # 5. ВЫЧИСЛЕНИЕ ПРОИЗВОДНЫХ ПРИЗНАКОВ
    print("Вычисление производных признаков...")
    current_time = pd.Timestamp.now()

    for item_id in item_stats_dict:
        stats = item_stats_dict[item_id]

        # Время с последнего взаимодействия
        if stats["item_last_ts"] > pd.Timestamp.min:
            stats["item_days_since_last"] = (current_time - stats["item_last_ts"]).days
        else:
            stats["item_days_since_last"] = 365

        # Время с первого взаимодействия
        if stats["item_first_ts"] < pd.Timestamp.max:
            stats["item_days_since_first"] = (
                current_time - stats["item_first_ts"]
            ).days
        else:
            stats["item_days_since_first"] = 365

        # Средний вес взаимодействия
        if stats["item_count"] > 0:
            stats["item_mean"] = stats["item_sum"] / stats["item_count"]
        else:
            stats["item_mean"] = 0

    print(f"Словарь товарных признаков построен. Записей: {len(item_stats_dict)}")
    return item_stats_dict


def build_user_item_features_dict(interactions_files, device="cuda"):
    """
    Создает словарь {(user_id, item_id): {feature_name: value}}.
    """
    print("Построение словаря пользователь-товарных признаков...")
    user_item_stats_dict = {}

    for f in tqdm(interactions_files, desc="Обработка взаимодействий"):
        df_chunk = pd.read_parquet(f)

        chunk_stats = df_chunk.groupby(["user_id", "item_id"]).agg(
            {
                "weight": ["count", "mean", "sum", "std", "max", "min"],
                "timestamp": ["max", "min"],
            }
        )

        for (user_id, item_id), stats in chunk_stats.iterrows():
            key = (user_id, item_id)
            if key not in user_item_stats_dict:
                user_item_stats_dict[key] = {
                    "ui_count": 0,
                    "ui_mean": 0,
                    "ui_sum": 0,
                    "ui_std": 0,
                    "ui_max": 0,
                    "ui_min": 0,
                    "ui_last_ts": pd.Timestamp.min,
                    "ui_first_ts": pd.Timestamp.max,
                }

            user_item_stats_dict[key]["ui_count"] += stats[("weight", "count")]
            user_item_stats_dict[key]["ui_sum"] += stats[("weight", "sum")]
            user_item_stats_dict[key]["ui_max"] = max(
                user_item_stats_dict[key]["ui_max"], stats[("weight", "max")]
            )
            user_item_stats_dict[key]["ui_min"] = min(
                user_item_stats_dict[key]["ui_min"], stats[("weight", "min")]
            )
            user_item_stats_dict[key]["ui_last_ts"] = max(
                user_item_stats_dict[key]["ui_last_ts"], stats[("timestamp", "max")]
            )
            user_item_stats_dict[key]["ui_first_ts"] = min(
                user_item_stats_dict[key]["ui_first_ts"], stats[("timestamp", "min")]
            )

    # ВЫЧИСЛЕНИЕ ПРОИЗВОДНЫХ ПРИЗНАКОВ
    print("Вычисление производных признаков...")
    current_time = pd.Timestamp.now()

    for key in user_item_stats_dict:
        stats = user_item_stats_dict[key]

        # Время с последнего взаимодействия
        if stats["ui_last_ts"] > pd.Timestamp.min:
            stats["ui_days_since_last"] = (current_time - stats["ui_last_ts"]).days
        else:
            stats["ui_days_since_last"] = 365

        # Время с первого взаимодействия
        if stats["ui_first_ts"] < pd.Timestamp.max:
            stats["ui_days_since_first"] = (current_time - stats["ui_first_ts"]).days
        else:
            stats["ui_days_since_first"] = 365

        # Средний вес взаимодействия
        if stats["ui_count"] > 0:
            stats["ui_mean"] = stats["ui_sum"] / stats["ui_count"]
        else:
            stats["ui_mean"] = 0

    print(
        f"Словарь пользователь-товарных признаков построен. Записей: {len(user_item_stats_dict)}"
    )
    return user_item_stats_dict


def build_category_features_dict(category_df, items_df):
    """
    Создает дополнительные признаки на основе категорий.
    """
    print("Построение категорийных признаков...")

    # Создаем маппинг товар -> категория
    item_to_cat = items_df.set_index("item_id")["catalogid"].to_dict()

    # Создаем маппинг категория -> уровень в иерархии
    cat_to_level = {}
    for _, row in category_df.iterrows():
        catalogid = row["catalogid"]
        # Уровень = количество родителей в иерархии
        level = (
            len(row["ids"]) - 1
        )  # -1 потому что ids включает сам catalogid и всех родителей
        cat_to_level[catalogid] = max(0, level)

    # Создаем словарь с категорийными признаками
    category_features_dict = {}
    for item_id, catalogid in item_to_cat.items():
        category_features_dict[item_id] = {
            "item_category": catalogid,
            "category_level": cat_to_level.get(catalogid, 0),
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

    print(f"=== РЕЖИМ МАСШТАБИРОВАНИЯ: {SCALING_STAGE.upper()} ===")
    print(f"Пользователей: {config['sample_users'] or 'все'}")
    print(f"Данных: {config['sample_fraction']*100}%")

    print("=== ЗАГРУЗКА ДАННЫХ ===")
    orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_ddf = (
        load_train_data()
    )
    orders_ddf, tracker_ddf, items_ddf = filter_data(orders_ddf, tracker_ddf, items_ddf)

    print("=== ЗАГРУЗКА ЭМБЕДДИНГОВ ===")
    # Загружаем эмбеддинги товаров - теперь возвращается только словарь
    embeddings_dict = load_and_process_embeddings(items_ddf)

    print("=== SPLIT ДАННЫХ ===")
    orders_df_full = orders_ddf.compute()
    train_orders_df, test_orders_df, cutoff_ts_per_user = train_test_split_by_time(
        orders_df_full, TEST_SIZE
    )

    print("=== ПОДГОТОВКА ВЗАИМОДЕЙСТВИЙ ===")
    interactions_files = prepare_interactions(
        train_orders_df, tracker_ddf, cutoff_ts_per_user, scale_days=30
    )

    print("=== ПОСЛЕДНИЕ ТОВАРЫ ===")
    batch_dir = "/home/root6/python/e_cup/rec_system/data/processed/prepare_interactions_batches"
    recent_items_map = build_recent_items_map_from_batches(batch_dir, recent_n=RECENT_N)

    print("=== ОБУЧЕНИЕ ALS ДЛЯ ПРИЗНАКОВ ===")
    model, user_map, item_map = train_als(
        interactions_files, n_factors=64, reg=1e-3, device="cuda"
    )
    inv_item_map = {v: k for k, v in item_map.items()}  # Создаем обратный маппинг
    popularity_s = compute_global_popularity(train_orders_df)
    popular_items = popularity_s.index.tolist()

    print("=== ПОСТРОЕНИЕ ДОПОЛНИТЕЛЬНЫХ ДАННЫХ ===")
    # Строим co-purchase map
    copurchase_map = build_copurchase_map(train_orders_df)

    # Строим категорийные маппинги
    items_df = items_ddf.compute()
    categories_df = categories_ddf.compute()
    item_to_cat, cat_to_items = build_category_maps(items_df, categories_df)

    # ЗАГРУЗКА/ПОДГОТОВКА ЭМБЕДДИНГОВ
    embeddings_dict = load_and_process_embeddings(items_ddf)

    print("=== ПРЕДВАРИТЕЛЬНЫЙ РАСЧЕТ ПРИЗНАКОВ ДЛЯ LGBM ===")
    user_features_dict = build_user_features_dict(interactions_files, orders_ddf)
    item_features_dict = build_item_features_dict(
        interactions_files, items_df, orders_ddf, embeddings_dict
    )
    user_item_features_dict = build_user_item_features_dict(interactions_files)

    print("=== ПОДГОТОВКА ДАННЫХ ДЛЯ LightGBM ===")
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
        sample_test_orders = test_orders_df  # все данные

    train_data = recommender.prepare_training_data(
        interactions_files,
        user_map,
        item_map,
        popularity_s,
        recent_items_map,
        sample_test_orders,
        sample_fraction=config["sample_fraction"],
    )

    # Разделяем на train/validation
    users = train_data["user_id"].unique()
    train_users, val_users = train_test_split(users, test_size=0.2, random_state=42)

    train_df = train_data[train_data["user_id"].isin(train_users)]
    val_df = train_data[train_data["user_id"].isin(val_users)]

    print(f"Размер train: {len(train_df)}, validation: {len(val_df)}")
    print(f"Признаки: {recommender.feature_columns[:20]}...")

    print("=== ОБУЧЕНИЕ LightGBM ===")
    model = recommender.train(train_df, val_df)

    print("=== ОЦЕНКА МОДЕЛИ ===")
    train_ndcg = recommender.evaluate(train_df)
    val_ndcg = recommender.evaluate(val_df)

    print(f"NDCG@100 train: {train_ndcg:.4f}")
    print(f"NDCG@100 val: {val_ndcg:.4f}")

    # Анализ важности признаков
    print("=== ВАЖНОСТЬ ПРИЗНАКОВ ===")
    feature_importance = pd.DataFrame(
        {
            "feature": recommender.feature_columns,
            "importance": recommender.model.feature_importance(),
        }
    )
    feature_importance = feature_importance.sort_values("importance", ascending=False)
    print(feature_importance.head(20))

    # СОХРАНЕНИЕ МОДЕЛИ И ВАЖНЫХ ДАННЫХ
    print("=== СОХРАНЕНИЕ МОДЕЛИ И ПРИЗНАКОВ ===")
    save_data = {
        "lgbm_model": recommender.model,
        "feature_columns": recommender.feature_columns,
        "als_model": model,
        "user_map": user_map,
        "item_map": item_map,
        "inv_item_map": inv_item_map,  # Сохраняем обратный маппинг!
        "popular_items": popular_items,
        "user_features_dict": user_features_dict,  # Сохраняем словари признаков
        "item_features_dict": item_features_dict,
        "user_item_features_dict": user_item_features_dict,
        "recent_items_map": recent_items_map,
        "copurchase_map": copurchase_map,
        "item_to_cat": item_to_cat,
    }
    with open(
        "/home/root6/python/e_cup/rec_system/src/models/lgbm_model_full.pkl", "wb"
    ) as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Обучение и подготовка данных завершены! Модель и признаки сохранены.")
    elapsed_time = timedelta(seconds=time.time() - start_time)
    print(f"Общее время выполнения: {elapsed_time}")

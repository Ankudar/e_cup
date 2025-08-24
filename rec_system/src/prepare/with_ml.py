import gc
import os
import pickle
import time
from collections import defaultdict
from datetime import timedelta
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
from tqdm.auto import tqdm

# tqdm интеграция с pandas
tqdm.pandas()


# -------------------- Загрузка данных --------------------
def load_train_data(max_parts=0, max_rows=0):
    """
    Загружаем parquet-файлы orders, tracker, items, categories_tree, test_users.
    Если колонка отсутствует в файле, пропускаем её.
    """
    paths = {
        "orders": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_orders_data/*/*.parquet",
        "tracker": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_tracker_data/*/*.parquet",
        "items": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_items_data/*.parquet",
        "categories": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train_final_categories_tree/*.parquet",
        "test_users": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_test_for_participants/test_for_participants/*.parquet",
    }

    columns_map = {
        "orders": ["item_id", "user_id", "created_timestamp", "last_status"],
        "tracker": ["item_id", "user_id", "timestamp", "action_type"],
        "items": ["item_id", "itemname", "fclip_embed", "catalogid"],
        "categories": ["catalogid", "catalogpath", "ids"],
        "test_users": ["user_id"],
    }

    def read_sample(path, columns=None, name=""):
        ddf = dd.read_parquet(path)
        # оставляем только колонки, которые реально есть
        if columns is not None:
            available_cols = [c for c in columns if c in ddf.columns]
            if not available_cols:
                print(f"{name}: ни одна из колонок {columns} не найдена, пропускаем")
                return dd.from_pandas(pd.DataFrame(), npartitions=1)
            ddf = ddf[available_cols]

        total_parts = ddf.npartitions
        if max_parts > 0:
            used_parts = min(total_parts, max_parts)
            ddf = ddf.partitions[:used_parts]
        else:
            used_parts = total_parts

        if max_rows > 0:
            actual_rows = min(max_rows, len(ddf))
            sample_df = ddf.head(actual_rows, compute=True)
            ddf = dd.from_pandas(sample_df, npartitions=1)

        count = ddf.shape[0].compute()
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

    # ====== Tracker ======
    print("... для tracker")
    tracker_ddf = tracker_ddf[["user_id", "item_id", "timestamp", "action_type"]]
    n_rows = tracker_ddf.size.compute() // 4  # 4 колонки

    for start in range(0, n_rows, batch_size):
        part = tracker_ddf.partitions[
            start // batch_size : (start + batch_size) // batch_size
        ]
        batch = part.compute()
        batch["timestamp"] = pd.to_datetime(batch["timestamp"])
        batch["cutoff"] = batch["user_id"].map(cutoff_ts_per_user)

        mask = batch["cutoff"].isna() | (batch["timestamp"] < batch["cutoff"])
        batch = batch.loc[mask]

        aw = batch["action_type"].map(action_weights).fillna(0)
        days_ago = (ref_time - batch["timestamp"]).dt.days.clip(lower=1)
        time_factor = np.log1p(days_ago / scale_days)
        batch = batch.assign(weight=aw * time_factor)[
            ["user_id", "item_id", "weight", "timestamp"]
        ]

        path = os.path.join(output_dir, f"tracker_batch_{start}.parquet")
        batch.to_parquet(path, index=False, engine="pyarrow")
        batch_files.append(path)
        del batch
        gc.collect()
        print(f"Сохранен tracker-батч {start}-{min(start+batch_size, n_rows)}")

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
def train_als(
    batch_files,
    factors=64,
    iterations=100,
    dtype=np.float32,
    random_state=42,
    device="cuda",
):
    """
    batch_files: список parquet файлов с взаимодействиями (user_id, item_id, weight)
    Возвращает: model, user_map, item_map
    """
    print("Подсчитываем уникальные user_id и item_id по всем батчам...")
    user_set, item_set = set(), set()
    for f in batch_files:
        df = pd.read_parquet(f)
        user_set.update(df["user_id"].unique())
        item_set.update(df["item_id"].unique())

    user_ids = sorted(user_set)
    item_ids = sorted(item_set)

    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {i: j for j, i in enumerate(item_ids)}

    print(f"Всего пользователей: {len(user_ids)}, товаров: {len(item_ids)}")

    # Собираем данные для COO матрицы
    rows, cols, values = [], [], []

    print("Формируем sparse матрицу по батчам...")
    for f in batch_files:
        df = pd.read_parquet(f)
        df["user_idx"] = df["user_id"].map(user_map).astype(np.int32)
        df["item_idx"] = df["item_id"].map(item_map).astype(np.int32)

        rows.extend(df["user_idx"].values)
        cols.extend(df["item_idx"].values)
        values.extend(df["weight"].astype(dtype).values)

    # Создаем torch sparse tensor на GPU
    rows_tensor = torch.tensor(rows, dtype=torch.long, device=device)
    cols_tensor = torch.tensor(cols, dtype=torch.long, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)

    sparse_tensor = torch.sparse_coo_tensor(
        torch.stack([rows_tensor, cols_tensor]),
        values_tensor,
        size=(len(user_ids), len(item_ids)),
        device=device,
    )

    # КРИТИЧЕСКИ ВАЖНО: coalesce tensor перед использованием
    sparse_tensor = sparse_tensor.coalesce()

    print(f"Общая COO матрица на GPU: {sparse_tensor.shape}")
    print(f"Ненулевых элементов: {sparse_tensor._nnz()}")

    # Обучаем ALS на GPU
    model = TorchALS(len(user_ids), len(item_ids), factors, device=device)
    model.fit(sparse_tensor, iterations=iterations)

    print("ALS обучение завершено")
    return model, user_map, item_map


# -------------------- User-Items CSR для recommend --------------------
def build_user_items_csr(
    batch_files, user_map, item_map, dtype=torch.float32, device="cuda"
):
    """
    batch_files: список parquet файлов с взаимодействиями
    Возвращает CSR матрицу user-items на GPU
    """
    print("Строим CSR матрицу по батчам на GPU...")

    # Создаем пустые тензоры для COO формата
    rows = []
    cols = []
    values = []

    for f in batch_files:
        df = pd.read_parquet(f)
        df["user_idx"] = df["user_id"].map(user_map).astype(np.int32)
        df["item_idx"] = df["item_id"].map(item_map).astype(np.int32)

        # Добавляем данные в COO формате
        rows.extend(df["user_idx"].values)
        cols.extend(df["item_idx"].values)
        values.extend(df["weight"].values.astype(np.float32))

    # Конвертируем в тензоры и переносим на GPU
    rows_tensor = torch.tensor(rows, dtype=torch.long, device=device)
    cols_tensor = torch.tensor(cols, dtype=torch.long, device=device)
    values_tensor = torch.tensor(values, dtype=dtype, device=device)

    # Создаем sparse матрицу на GPU
    sparse_tensor = torch.sparse_coo_tensor(
        torch.stack([rows_tensor, cols_tensor]),
        values_tensor,
        size=(len(user_map), len(item_map)),
        device=device,
    )

    print(f"CSR матрица построена на GPU: {sparse_tensor.shape}")
    return sparse_tensor


def build_copurchase_map(train_orders_df, min_co_items=2, top_n=10, device="cuda"):
    """
    Быстрая версия: строим словарь совместных покупок на GPU.
    """
    print("Строим co-purchase матрицу на GPU...")

    # Группируем по (user_id, timestamp) и оставляем только корзины с min_co_items
    baskets = (
        train_orders_df.groupby(["user_id", "created_timestamp"])["item_id"]
        .apply(list)
        .tolist()
    )
    baskets = [b for b in baskets if len(b) >= min_co_items]

    # Словарь {item_id -> index} для построения матрицы
    unique_items = train_orders_df["item_id"].unique()
    item2idx = {it: i for i, it in enumerate(unique_items)}
    idx2item = {i: it for it, i in item2idx.items()}
    n_items = len(unique_items)

    # Sparse co-occurrence матрица
    co_matrix = torch.zeros((n_items, n_items), dtype=torch.float32, device=device)

    for items in baskets:
        idxs = torch.tensor([item2idx[it] for it in items], device=device)
        # Строим матрицу смежности корзины (ones)
        mask = torch.ones((len(idxs), len(idxs)), device=device)
        mask.fill_diagonal_(0.0)  # не считаем item→item
        weight = 1.0 / len(items)
        co_matrix[idxs.unsqueeze(1), idxs.unsqueeze(0)] += mask * weight

    # Нормализация построчно
    row_sums = co_matrix.sum(dim=1, keepdim=True).clamp(min=1e-9)
    norm_matrix = co_matrix / row_sums

    # Формируем финальный словарь топ-N для каждого item
    final_copurchase = {}
    for i in range(n_items):
        row = norm_matrix[i]
        if row.sum() > 0:
            topk_vals, topk_idx = torch.topk(row, k=min(top_n, n_items))
            final_copurchase[idx2item[i]] = [
                (idx2item[j.item()], v.item())
                for j, v in zip(topk_idx, topk_vals)
                if v.item() > 0
            ]

    print(f"Co-purchase словарь построен для {len(final_copurchase)} товаров")
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


# -------------------- Recommendations (ALS + recent + similar + popular) --------------------
def generate_and_save_recommendations_iter(
    user_batch,
    train_orders_df,
    user_map,
    item_map,
    inv_item_map,
    user_factors,
    item_factors,
    user_items_sparse,
    als_weight=0.5,
    co_weight=0.3,
    cat_weight=0.2,
    top_k=100,
    recent_n=5,
    similar_top_n_seed=20,
    copurchase_map=None,
    cat_to_items=None,
    popular_items=None,
):
    """
    Кандидаты с ускорением на PyTorch (ALS + similarity батчами).
    """
    device = user_factors.device
    als_recommendations = {}
    co_recommendations = {}
    cat_recommendations = {}
    pop_recommendations = {}
    sim_recommendations = {}

    # === ALS ===
    valid_user_indices = [user_map[u] for u in user_batch if u in user_map]
    if valid_user_indices:
        user_embeddings = user_factors[valid_user_indices]  # (B, d)
        with torch.no_grad():
            scores = torch.matmul(user_embeddings, item_factors.T)  # (B, N)

            # маска просмотренных
            user_interactions = user_items_sparse[valid_user_indices].to_dense()
            scores[user_interactions > 0] = -float("inf")

            top_scores, top_indices = torch.topk(scores, top_k * 2, dim=1)

        for j, u in enumerate(user_batch):
            if u in user_map:
                als_items = [
                    (inv_item_map[int(idx)], float(score * als_weight))
                    for idx, score in zip(top_indices[j].cpu(), top_scores[j].cpu())
                    if score > -float("inf") and int(idx) in inv_item_map
                ]
                als_recommendations[u] = als_items

    # === Co-purchase ===
    if copurchase_map is not None:
        for u in user_batch:
            user_orders = train_orders_df[train_orders_df.user_id == u]
            recent_items = (
                user_orders.sort_values("created_timestamp")["item_id"]
                .drop_duplicates()
                .tolist()[-recent_n:]
            )
            co_items = {}
            for rit in recent_items:
                if rit in copurchase_map:
                    for cit, w in copurchase_map[rit]:
                        co_items[cit] = co_items.get(cit, 0) + w * co_weight
            co_recommendations[u] = sorted(co_items.items(), key=lambda x: -x[1])[
                :top_k
            ]

    # === Похожие товары (батчем) ===
    for u in user_batch:
        user_orders = train_orders_df[train_orders_df.user_id == u]
        recent_items = (
            user_orders.sort_values("created_timestamp")["item_id"]
            .drop_duplicates()
            .tolist()[-recent_n:]
        )
        recent_indices = [item_map[it] for it in recent_items if it in item_map]
        sim_items = []
        if recent_indices:
            rit_embeddings = item_factors[recent_indices]  # (R, d)
            with torch.no_grad():
                similarities = torch.matmul(item_factors, rit_embeddings.T)  # (N, R)
                topk_vals, topk_idx = torch.topk(
                    similarities, similar_top_n_seed + 1, dim=0
                )
            for col in range(len(recent_indices)):
                for idx in topk_idx[1:, col].cpu().numpy():
                    if int(idx) in inv_item_map:
                        sim_items.append(inv_item_map[int(idx)])
            sim_items = list(dict.fromkeys(sim_items))[:top_k]
        sim_recommendations[u] = [(it, 1.0) for it in sim_items]

    # === Категории ===
    if cat_to_items is not None:
        for u in user_batch:
            user_orders = train_orders_df[train_orders_df.user_id == u]
            recent_items = (
                user_orders.sort_values("created_timestamp")["item_id"]
                .drop_duplicates()
                .tolist()[-recent_n:]
            )
            cat_items = {}
            for rit in recent_items:
                for cit in cat_to_items.get(rit, []):
                    cat_items[cit] = cat_items.get(cit, 0) + cat_weight
            cat_recommendations[u] = sorted(cat_items.items(), key=lambda x: -x[1])[
                :top_k
            ]

    # === Популярные ===
    if popular_items is not None:
        pop_recommendations = {
            u: [(it, 1.0) for it in popular_items[:top_k]] for u in user_batch
        }

    # === Объединение ===
    final_recommendations = {}
    for u in user_batch:
        combined = defaultdict(float)
        for source in [
            als_recommendations,
            co_recommendations,
            sim_recommendations,
            cat_recommendations,
            pop_recommendations,
        ]:
            if u in source:
                for it, score in source[u]:
                    combined[it] += score
        final_recommendations[u] = sorted(combined.items(), key=lambda x: -x[1])[:top_k]

    return final_recommendations


def evaluate_ndcg_iter_optimized(
    model,
    user_map,
    item_map,
    user_items_csr,
    test_df,
    popularity_s,
    users_df,
    interactions_files,
    recent_items_map=None,
    top_k=100,
    recent_n=5,
    similar_top_n_seed=20,
    blend_sim_beta=0.3,
):
    import pandas as pd
    from tqdm import tqdm

    recent_items_map = recent_items_map or {}
    inv_item_map = {int(v): int(k) for k, v in item_map.items()}
    popular_items = popularity_s.index.tolist()
    popular_set = set(popular_items)

    # ======== Предварительно создаем ground truth ========
    gt = test_df.groupby("user_id")["item_id"].apply(set).to_dict()

    # ======== Предзагрузка всех интеракций ========
    user_interactions_dict = {}
    for f in interactions_files:
        df = pd.read_parquet(f)
        for uid, df_user in df.groupby("user_id"):
            if uid in user_interactions_dict:
                user_interactions_dict[uid] = pd.concat(
                    [user_interactions_dict[uid], df_user], ignore_index=True
                )
            else:
                user_interactions_dict[uid] = df_user

    ndcg_sum = 0.0
    n_users = 0

    for user_id in tqdm(users_df["user_id"], desc="Evaluate NDCG iter"):
        try:
            user_idx_int = user_map.get(user_id)
            user_tracker = user_interactions_dict.get(
                user_id,
                pd.DataFrame(columns=["user_id", "item_id", "weight", "timestamp"]),
            )
            user_bought_items = set(user_tracker[user_tracker["weight"] > 0]["item_id"])
            tracker_scored = (
                user_tracker[["item_id", "weight"]].drop_duplicates().values.tolist()
            )

            # ======== ALS рекомендации ========
            als_scored = []
            if user_idx_int is not None and user_idx_int < user_items_csr.shape[0]:
                try:
                    rec = model.recommend(
                        userid=user_idx_int,
                        user_items=user_items_csr[user_idx_int],
                        N=top_k * 2,
                        filter_already_liked_items=True,
                        recalculate_user=True,
                    )
                    als_items = [
                        (inv_item_map[int(r[0])], float(r[1]))
                        for r in rec
                        if len(r) == 2 and int(r[0]) in inv_item_map
                    ]
                    als_scored = [
                        (it, score * 0.5)
                        for it, score in als_items
                        if it not in user_bought_items
                    ]
                except Exception as e:
                    print(f"Ошибка ALS для пользователя {user_id}: {e}")

            # ======== Последние товары ========
            recent_items = recent_items_map.get(user_id, [])
            recent_scored = [
                (it, 1.0)
                for it in recent_items[:recent_n]
                if it not in user_bought_items
            ]

            # ======== Похожие товары ========
            sim_items = []
            for rit in recent_items:
                if rit in item_map:
                    try:
                        sims = model.similar_items(
                            int(item_map[rit]), N=similar_top_n_seed
                        )
                        sim_items.extend(
                            [
                                inv_item_map[int(si[0])]
                                for si in sims
                                if len(si) == 2 and int(si[0]) in inv_item_map
                            ]
                        )
                    except Exception:
                        continue
            sim_items = list(dict.fromkeys(sim_items))
            sim_quota = int(top_k * blend_sim_beta)
            sim_scored = [
                (it, 0.5) for it in sim_items[:sim_quota] if it not in user_bought_items
            ]

            # ======== Популярные товары ========
            min_score = min((s for _, s in als_scored), default=0.1)
            pop_scored = [
                (it, min_score * 0.1)
                for it in popular_items
                if it not in user_bought_items
            ]

            # ======== Объединяем кандидатов ========
            all_candidates = (
                tracker_scored + als_scored + recent_scored + sim_scored + pop_scored
            )
            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)

            seen, final = set(), []
            for it, _ in all_candidates:
                if it not in seen:
                    final.append(it)
                    seen.add(it)
                if len(final) >= top_k:
                    break
            recommended = final if final else popular_items[:top_k]

            # ======== NDCG ========
            user_gt = gt.get(user_id, set())
            ndcg_sum += ndcg_at_k(recommended, user_gt, k=top_k)
            n_users += 1

        except Exception as e:
            print(f"Ошибка для пользователя {user_id}: {e}")
            continue

    return ndcg_sum / n_users if n_users > 0 else 0.0


# -------------------- Строим словарь "какие товары похожи на какой" --------------------
def get_similar_by_category_tree(items_df, categories_df, top_n=50):
    """
    Строит словарь похожих товаров через иерархию категорий
    """
    # Группировка: категория -> список товаров
    catalog_items = items_df.groupby("catalogid")["item_id"].apply(list).to_dict()
    # Иерархия категорий
    cat_tree = categories_df.set_index("catalogid")["ids"].to_dict()

    # Предрассчитаем "расширенные" категории сразу
    extended_cat_items = {}
    for cat_id, items in catalog_items.items():
        all_items = set(items)
        if cat_id in cat_tree:
            for parent_cat in cat_tree[cat_id]:
                all_items.update(catalog_items.get(parent_cat, []))
        extended_cat_items[cat_id] = list(all_items)

    # Для каждого товара ищем похожие
    similar_map = {}
    for item, cat in items_df[["item_id", "catalogid"]].values:
        sim_items = extended_cat_items.get(cat, [])
        if sim_items:
            similar_map[item] = [x for x in sim_items if x != item][:top_n]
        else:
            similar_map[item] = []
    return similar_map


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


def evaluate_ndcg(recommendations, test_df, k=100, device="cuda"):
    """
    Считаем средний NDCG@k по всем пользователям
    recommendations: dict[user_id -> список рекомендованных item_id]
    test_df: DataFrame с (user_id, item_id)
    """
    print("Считаем NDCG...")

    # Собираем ground truth в dict[user -> set(items)]
    gt = defaultdict(set)
    for r in test_df.itertuples():
        gt[r.user_id].add(r.item_id)

    ndcg_scores = []
    for uid, recs in recommendations.items():
        ndcg_scores.append(ndcg_at_k(recs, gt.get(uid, set()), k=k, device=device))

    print("NDCG рассчитан")
    return (
        float(torch.tensor(ndcg_scores, device=device).mean().item())
        if ndcg_scores
        else 0.0
    )


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
        self, n_users, n_items, n_factors, reg=0.1, dtype=torch.float32, device="cuda"
    ):
        super().__init__()
        self.user_factors = nn.Parameter(
            torch.randn(n_users, n_factors, dtype=dtype, device=device) * 0.01
        )
        self.item_factors = nn.Parameter(
            torch.randn(n_items, n_factors, dtype=dtype, device=device) * 0.01
        )
        self.reg = reg
        self.device = device
        self.to(device)

    def forward(self, user, item):
        return (self.user_factors[user] * self.item_factors[item]).sum(1)

    def fit(
        self,
        sparse_tensor,
        iterations=100,
        lr=0.01,
        show_progress=True,
        sample_ratio=0.1,
    ):
        """Обучение ALS с семплированием для больших данных"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.reg)

        if not sparse_tensor.is_coalesced():
            sparse_tensor = sparse_tensor.coalesce()

        users_coo, items_coo = sparse_tensor.indices()
        values = sparse_tensor.values()
        n_interactions = len(values)

        for epoch in range(iterations):
            # Семплируем часть взаимодействий
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

            if show_progress and (epoch % 10 == 0 or epoch == iterations - 1):
                print(f"Iteration {epoch}, Loss: {total_loss.item():.6f}")

    def recommend(self, user_ids, user_items=None, N=10, filter_already_liked=True):
        """Рекомендации для пользователей"""
        if isinstance(user_ids, int):
            user_ids = [user_ids]

        user_embeddings = self.user_factors[user_ids]  # (B, factors)
        scores = torch.matmul(user_embeddings, self.item_factors.T)  # (B, n_items)

        if filter_already_liked and user_items is not None:
            # Маскируем уже просмотренные товары
            scores[user_items[user_ids].to_dense() > 0] = -float("inf")

        # Получаем топ-N товаров
        top_scores, top_indices = torch.topk(scores, k=min(N, scores.size(1)), dim=1)

        return [
            (indices.cpu().numpy(), scores.cpu().numpy())
            for indices, scores in zip(top_indices, top_scores)
        ]


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
    ):
        """
        Подготовка данных для LightGBM с ограничением негативных примеров (≤3 на пользователя)
        """
        print("Подготовка данных для LightGBM...")

        # Ограничиваем данные
        test_orders_df = test_orders_df.sample(frac=sample_fraction, random_state=42)

        # Загружаем все взаимодействия
        print("Загрузка всех взаимодействий для признаков...")
        all_interactions = []
        for f in tqdm(interactions_files, desc="Загрузка взаимодействий"):
            df = pd.read_parquet(f)
            all_interactions.append(df)
        interactions_df = pd.concat(all_interactions, ignore_index=True)

        # Временные метки
        interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
        max_timestamp = interactions_df["timestamp"].max()

        # Позитивные пары
        positive_samples = [
            {"user_id": row["user_id"], "item_id": row["item_id"], "target": 1}
            for _, row in tqdm(test_orders_df.iterrows(), desc="Позитивные примеры")
        ]
        positive_df = pd.DataFrame(positive_samples)

        # Негативные примеры
        print("Улучшенное негативное сэмплирование...")
        negative_samples = []

        user_items = interactions_df.groupby("user_id")["item_id"].apply(set).to_dict()
        all_items_set = set(item_map.keys())
        popularity_series = popularity_s.copy()

        for user_id in tqdm(positive_df["user_id"].unique(), desc="Негативные сэмплы"):
            user_positives = set(
                positive_df[positive_df["user_id"] == user_id]["item_id"]
            )
            user_interacted = user_items.get(user_id, set())

            # Доступные негативы
            available_negatives = list(all_items_set - user_interacted - user_positives)

            if available_negatives:
                n_negatives = min(5, len(available_negatives))

                available_in_popularity = [
                    item
                    for item in available_negatives
                    if item in popularity_series.index
                ]

                if available_in_popularity:
                    popular_negatives = (
                        popularity_series[available_in_popularity]
                        .nlargest(min(n_negatives, len(available_in_popularity)))
                        .index.tolist()
                    )
                else:
                    popular_negatives = []

                remaining_negatives = list(
                    set(available_negatives) - set(popular_negatives)
                )
                if remaining_negatives:
                    random_negatives = np.random.choice(
                        remaining_negatives,
                        min(
                            n_negatives - len(popular_negatives),
                            len(remaining_negatives),
                        ),
                        replace=False,
                    ).tolist()
                else:
                    random_negatives = []

                negative_items = popular_negatives + random_negatives

                for item_id in negative_items:
                    negative_samples.append(
                        {"user_id": user_id, "item_id": item_id, "target": 0}
                    )

        negative_df = pd.DataFrame(negative_samples)

        # Объединение
        train_data = pd.concat([positive_df, negative_df], ignore_index=True)

        # Богатые признаки
        train_data = self._add_rich_features(
            train_data,
            interactions_df,
            popularity_s,
            recent_items_map,
            user_map,
            item_map,
            max_timestamp,
        )

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

        # === FCLIP ЭМБЕДДИНГИ ===
        if self.external_embeddings_dict:
            print("Добавление внешних эмбеддингов...")
            sample_embedding = next(iter(self.external_embeddings_dict.values()))
            for i in range(min(10, len(sample_embedding))):
                data[f"fclip_embed_{i}"] = data["item_id"].map(
                    lambda x: (
                        self.external_embeddings_dict.get(x, np.zeros(1))[i]
                        if x in self.external_embeddings_dict
                        else 0.0
                    )
                )

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


def load_and_process_embeddings(
    items_ddf, embedding_column="fclip_embed", device="cuda"
):
    """
    Загрузка и обработка эмбеддингов товаров
    """
    print("Загрузка эмбеддингов товаров...")

    with ProgressBar():
        items_df = items_ddf.compute()

    if embedding_column not in items_df.columns:
        print(f"Колонка {embedding_column} не найдена в items данных")
        return None

    embeddings_dict = {}
    valid_items = []

    for idx, row in tqdm(
        items_df.iterrows(), desc="Обработка эмбеддингов", total=len(items_df)
    ):
        item_id = row["item_id"]
        embedding_str = row[embedding_column]

        if isinstance(embedding_str, str) and pd.notna(embedding_str):
            try:
                embedding = np.fromstring(
                    embedding_str.strip("[]"), sep=" ", dtype=np.float32
                )
                if embedding.size > 0:
                    # Сохраняем как numpy array для совместимости с LightGBM
                    embeddings_dict[item_id] = embedding
                    valid_items.append(item_id)
            except Exception as e:
                print(f"Ошибка обработки эмбеддинга для товара {item_id}: {e}")
                continue
        elif isinstance(embedding_str, (list, np.ndarray)):
            arr = np.array(embedding_str, dtype=np.float32)
            if arr.size > 0:
                embeddings_dict[item_id] = arr
                valid_items.append(item_id)
        elif hasattr(embedding_str, "__array__"):
            # Для случаев, когда это уже тензор или array-like объект
            try:
                arr = np.array(embedding_str, dtype=np.float32)
                if arr.size > 0:
                    embeddings_dict[item_id] = arr
                    valid_items.append(item_id)
            except Exception as e:
                print(f"Ошибка конвертации эмбеддинга для товара {item_id}: {e}")
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
    model, user_map, item_map = train_als(interactions_files)
    popularity_s = compute_global_popularity(train_orders_df)

    print("=== ПОСТРОЕНИЕ ДОПОЛНИТЕЛЬНЫХ ДАННЫХ ===")
    # Строим co-purchase map
    copurchase_map = build_copurchase_map(train_orders_df)

    # Строим категорийные маппинги
    items_df = items_ddf.compute()
    categories_df = categories_ddf.compute()
    item_to_cat, cat_to_items = build_category_maps(items_df, categories_df)

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

    print("=== СОХРАНЕНИЕ МОДЕЛИ ===")
    os.makedirs("/home/root6/python/e_cup/rec_system/src/models", exist_ok=True)
    with open(
        "/home/root6/python/e_cup/rec_system/src/models/lgbm_model.pkl", "wb"
    ) as f:
        pickle.dump(
            {
                "model": recommender.model,
                "feature_columns": recommender.feature_columns,
                "user_map": user_map,
                "item_map": item_map,
                "popularity_s": popularity_s,
                "embeddings_dict": embeddings_dict,
            },
            f,
        )

    print("Обучение завершено! Модель сохранена.")
    elapsed_time = timedelta(seconds=time.time() - start_time)
    print(f"Время подготовки модели: {elapsed_time}")

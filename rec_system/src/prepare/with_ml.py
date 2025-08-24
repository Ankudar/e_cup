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
    Фильтруем: берём только доставленные заказы и ограниченные действия в tracker.
    """
    print("Фильтрация данных...")
    orders_ddf = orders_ddf[orders_ddf["last_status"] == "delivered_orders"]
    allowed_actions = ["page_view", "favorite", "to_cart"]
    tracker_ddf = tracker_ddf[tracker_ddf["action_type"].isin(allowed_actions)]
    print("Фильтрация завершена")
    return orders_ddf, tracker_ddf, items_ddf


# -------------------- Train/Test split по времени --------------------
def train_test_split_by_time(orders_df, test_size=0.2):
    """
    Быстрый temporal split на train/test для каждого пользователя без Python loop.
    Возвращаем train_df, test_df и cutoff_ts_per_user.
    """
    print("Делаем train/test split...")
    orders_df = orders_df.copy()
    orders_df["created_timestamp"] = pd.to_datetime(orders_df["created_timestamp"])

    # сортируем
    orders_df = orders_df.sort_values(["user_id", "created_timestamp"])

    # для каждого пользователя считаем размер теста
    user_counts = orders_df.groupby("user_id").size()
    test_counts = (user_counts * test_size).apply(lambda x: max(1, int(x)))

    # индекс последнего train
    orders_df["cumcount"] = orders_df.groupby("user_id").cumcount()
    orders_df["test_count"] = orders_df["user_id"].map(test_counts)

    mask_test = orders_df["cumcount"] >= (
        orders_df.groupby("user_id")["cumcount"].transform("max")
        + 1
        - orders_df["test_count"]
    )
    train_df = orders_df[~mask_test].drop(columns=["cumcount", "test_count"])
    test_df = orders_df[mask_test].drop(columns=["cumcount", "test_count"])

    # cutoff timestamp для каждого пользователя
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
        batch_orders = train_orders_df.iloc[start : start + batch_size].copy()
        batch_orders["days_ago"] = (
            ref_time - batch_orders["created_timestamp"]
        ).dt.days.clip(lower=1)
        batch_orders["time_factor"] = np.log1p(batch_orders["days_ago"] / scale_days)
        batch_orders["weight"] = 5.0 * batch_orders["time_factor"]
        batch_orders = batch_orders.rename(columns={"created_timestamp": "timestamp"})
        batch_orders = batch_orders[["user_id", "item_id", "weight", "timestamp"]]

        orders_file = os.path.join(output_dir, f"orders_batch_{start}.parquet")
        batch_orders.to_parquet(orders_file, index=False, engine="pyarrow")
        batch_files.append(orders_file)
        del batch_orders
        gc.collect()
        print(f"Сохранен orders-батч строк {start}-{min(start+batch_size, n_rows)}")

    # ====== Tracker ======
    print("... для tracker")
    tracker_ddf = tracker_ddf[["user_id", "item_id", "timestamp", "action_type"]]
    n_rows = tracker_ddf.shape[0].compute()  # число строк

    for start in range(0, n_rows, batch_size):
        batch_ddf = tracker_ddf.partitions[
            start // batch_size : (start + batch_size) // batch_size
        ]
        batch_tracker = batch_ddf.compute()
        batch_tracker["timestamp"] = pd.to_datetime(batch_tracker["timestamp"])
        batch_tracker["aw"] = batch_tracker["action_type"].map(action_weights).fillna(0)
        batch_tracker["cutoff"] = batch_tracker["user_id"].map(cutoff_ts_per_user)
        batch_tracker = batch_tracker[
            (batch_tracker["cutoff"].isna())
            | (batch_tracker["timestamp"] < batch_tracker["cutoff"])
        ]
        batch_tracker["days_ago"] = (
            ref_time - batch_tracker["timestamp"]
        ).dt.days.clip(lower=1)
        batch_tracker["time_factor"] = np.log1p(batch_tracker["days_ago"] / scale_days)
        batch_tracker["weight"] = batch_tracker["aw"] * batch_tracker["time_factor"]
        batch_tracker = batch_tracker[["user_id", "item_id", "weight", "timestamp"]]

        tracker_file = os.path.join(output_dir, f"tracker_batch_{start}.parquet")
        batch_tracker.to_parquet(tracker_file, index=False, engine="pyarrow")
        batch_files.append(tracker_file)
        del batch_tracker
        gc.collect()
        print(f"Сохранен tracker-батч строк {start}-{min(start+batch_size, n_rows)}")

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
    batch_files, factors=64, iterations=100, dtype=np.float32, random_state=42
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

    coo_batches = []

    print("Формируем sparse матрицу по батчам...")
    for f in batch_files:
        df = pd.read_parquet(f)
        df["user_idx"] = df["user_id"].map(user_map).astype(np.int32)
        df["item_idx"] = df["item_id"].map(item_map).astype(np.int32)
        coo = coo_matrix(
            (df["weight"].astype(dtype), (df["user_idx"], df["item_idx"])),
            shape=(len(user_ids), len(item_ids)),
        )
        coo_batches.append(coo)

    sparse_interactions = vstack(coo_batches)
    print(f"Общая COO матрица: {sparse_interactions.shape}")

    model = AlternatingLeastSquares(
        factors=factors, iterations=iterations, random_state=random_state
    )

    model.fit(sparse_interactions.T, show_progress=True)
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


def build_copurchase_map(train_orders_df, min_co_items=2, top_n=10):
    """
    Улучшенная версия: строим словарь совместных покупок с весами
    """
    print("Строим улучшенный словарь co-purchase...")
    copurchase_dict = defaultdict(lambda: defaultdict(float))

    # Группируем покупки по пользователям и времени
    grouped = train_orders_df.groupby(["user_id", "created_timestamp"])[
        "item_id"
    ].apply(list)

    for items in grouped:
        if len(items) >= min_co_items:
            # Взвешиваем по количеству товаров в корзине (чем больше корзина, тем слабее связи)
            weight = 1.0 / len(items)
            for i, item1 in enumerate(items):
                for j, item2 in enumerate(items):
                    if i != j:
                        copurchase_dict[item1][item2] += weight

    # Нормализуем и оставляем топ-N
    final_copurchase = {}
    for item, co_items in copurchase_dict.items():
        total = sum(co_items.values())
        # Нормализуем и сортируем
        sorted_items = sorted(
            co_items.items(), key=lambda x: x[1] / total, reverse=True
        )
        final_copurchase[item] = [
            (it, weight / total) for it, weight in sorted_items[:top_n]
        ]

    print(f"Co-purchase словарь построен для {len(final_copurchase)} товаров")
    return final_copurchase


def build_category_maps(items_df, categories_df):
    """
    Строим маппинги товаров к категориям и категорий к товарам
    """
    print("Построение категорийных маппингов...")

    # Товар -> категория
    item_to_cat = items_df.set_index("item_id")["catalogid"].to_dict()

    # Категория -> список товаров
    cat_to_items = items_df.groupby("catalogid")["item_id"].apply(list).to_dict()

    # Иерархия категорий
    cat_tree = categories_df.set_index("catalogid")["ids"].to_dict()

    # Расширяем cat_to_items с учетом иерархии
    extended_cat_to_items = {}
    for cat_id, items_list in cat_to_items.items():
        all_items = set(items_list)
        if cat_id in cat_tree:
            for parent_cat in cat_tree[cat_id]:
                if parent_cat in cat_to_items:
                    all_items.update(cat_to_items[parent_cat])
        extended_cat_to_items[cat_id] = list(all_items)

    return item_to_cat, extended_cat_to_items


# -------------------- Recommendations (ALS + recent + similar + popular) --------------------
def generate_and_save_recommendations_iter(
    model,
    user_map,
    item_map,
    user_items_csr,
    interactions_files,
    popularity_s,
    users_df,
    recent_items_map=None,
    copurchase_map=None,
    top_k=100,
    recent_n=5,
    similar_top_n_seed=20,
    blend_sim_beta=0.3,
    blend_cat_beta=0.3,
    batch_size=10_000,
    filename="/home/root6/python/e_cup/rec_system/result/submission.csv",
    item_to_cat=None,
    cat_to_items=None,
    device="cuda",
):
    recent_items_map = recent_items_map or {}
    copurchase_map = copurchase_map or {}
    inv_item_map = {int(v): int(k) for k, v in item_map.items()}
    popular_items = popularity_s.index.tolist()

    # Переносим модель и данные на GPU
    user_factors = torch.tensor(model.user_factors, device=device, dtype=torch.float32)
    item_factors = torch.tensor(model.item_factors, device=device, dtype=torch.float32)

    # Конвертируем CSR в torch sparse tensor на GPU
    user_items_sparse = torch.sparse_csr_tensor(
        user_items_csr.indptr,
        user_items_csr.indices,
        user_items_csr.data,
        size=user_items_csr.shape,
        dtype=torch.float32,
        device=device,
    )

    # Предзагрузка всех интеракций в память для быстрого доступа
    print("Предзагрузка интеракций пользователей...")
    user_interactions_dict = {}
    for f in tqdm(interactions_files, desc="Загрузка файлов интеракций"):
        df = pd.read_parquet(f)
        for uid, group in df.groupby("user_id"):
            if uid not in user_interactions_dict:
                user_interactions_dict[uid] = group
            else:
                user_interactions_dict[uid] = pd.concat(
                    [user_interactions_dict[uid], group]
                )

    first_run = True
    if os.path.exists(filename):
        os.remove(filename)

    batch_rows = []

    # Векторизованная обработка пользователей
    user_ids_list = users_df["user_id"].tolist()

    for i in tqdm(
        range(0, len(user_ids_list), batch_size), desc="Обработка пользователей батчами"
    ):
        user_batch = user_ids_list[i : i + batch_size]

        try:
            # ======== Пакетная обработка ALS рекомендаций на GPU ========
            user_indices = [user_map.get(uid, -1) for uid in user_batch]
            valid_user_mask = [
                idx != -1 and idx < user_items_sparse.shape[0] for idx in user_indices
            ]
            valid_user_indices = [
                idx for idx, valid in zip(user_indices, valid_user_mask) if valid
            ]

            als_recommendations = {}
            if valid_user_indices:
                # Получаем эмбеддинги пользователей
                user_embeddings = user_factors[valid_user_indices]

                # Вычисляем scores на GPU
                with torch.no_grad():
                    scores = torch.matmul(item_factors, user_embeddings.T).T

                    # Создаем маску для уже просмотренных товаров
                    user_interactions_mask = torch.zeros(
                        (len(valid_user_indices), item_factors.shape[0]),
                        device=device,
                        dtype=torch.bool,
                    )

                    for j, user_idx in enumerate(valid_user_indices):
                        user_interactions = user_items_sparse[user_idx].to_dense()
                        user_interactions_mask[j] = user_interactions > 0

                    # Маскируем уже просмотренные
                    scores[user_interactions_mask] = -float("inf")

                    # Берем топ-K для каждого пользователя
                    top_scores, top_indices = torch.topk(scores, top_k * 2, dim=1)

                    for j, user_idx in enumerate(valid_user_indices):
                        user_id = [
                            uid
                            for uid, idx in zip(user_batch, user_indices)
                            if idx == user_idx
                        ][0]
                        als_items = [
                            (inv_item_map[int(idx)], float(score * 0.5))
                            for idx, score in zip(
                                top_indices[j].cpu(), top_scores[j].cpu()
                            )
                            if score > -float("inf") and int(idx) in inv_item_map
                        ]
                        als_recommendations[user_id] = als_items

            # Обрабатываем каждого пользователя в батче
            for user_id in user_batch:
                try:
                    # ======== Загружаем интеракции пользователя из предзагруженного словаря ========
                    user_tracker = user_interactions_dict.get(
                        user_id,
                        pd.DataFrame(
                            columns=["user_id", "item_id", "weight", "timestamp"]
                        ),
                    )

                    user_bought_items = set(
                        user_tracker[user_tracker["weight"] > 0]["item_id"]
                    )
                    tracker_scored = (
                        user_tracker[["item_id", "weight"]]
                        .drop_duplicates()
                        .values.tolist()
                    )

                    # ======== ALS рекомендации из пакетного результата ========
                    als_scored = als_recommendations.get(user_id, [])
                    als_scored = [
                        (it, score)
                        for it, score in als_scored
                        if it not in user_bought_items
                    ]

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
                                # Векторизованный поиск похожих товаров на GPU
                                rit_idx = item_map[rit]
                                rit_embedding = item_factors[rit_idx].unsqueeze(0)

                                with torch.no_grad():
                                    similarities = torch.matmul(
                                        item_factors, rit_embedding.T
                                    ).squeeze()
                                    top_similar = torch.topk(
                                        similarities, similar_top_n_seed + 1
                                    )

                                    similar_ids = [
                                        inv_item_map[int(idx)]
                                        for idx in top_similar.indices[
                                            1:
                                        ].cpu()  # пропускаем сам товар
                                        if int(idx) in inv_item_map
                                    ]
                                    sim_items.extend(similar_ids)

                            except Exception:
                                continue

                    sim_items = list(dict.fromkeys(sim_items))
                    sim_quota = int(top_k * blend_sim_beta)
                    sim_scored = [
                        (it, 0.5)
                        for it in sim_items[:sim_quota]
                        if it not in user_bought_items
                    ]

                    # ======== Товары из категорий последних покупок ========
                    category_scored = []
                    if item_to_cat and cat_to_items:
                        for rit in recent_items:
                            cat = item_to_cat.get(rit)
                            if cat:
                                cat_items = [
                                    i
                                    for i in cat_to_items[cat]
                                    if i != rit and i not in user_bought_items
                                ]
                                quota = int(top_k * blend_cat_beta)
                                category_scored.extend(
                                    [(i, 0.8) for i in cat_items[:quota]]
                                )

                    # ======== Co-purchase рекомендации ========
                    copurchase_scored = []
                    for rit in recent_items:
                        for co_item in copurchase_map.get(rit, []):
                            if co_item not in user_bought_items:
                                copurchase_scored.append((co_item, 0.8))

                    # ======== Популярные товары ========
                    min_score = min((s for _, s in als_scored), default=0.1)
                    pop_scored = [
                        (it, min_score * 0.1)
                        for it in popular_items
                        if it not in user_bought_items
                    ]

                    # ======== Объединяем все кандидаты ========
                    all_candidates = (
                        tracker_scored
                        + als_scored
                        + recent_scored
                        + sim_scored
                        + category_scored
                        + copurchase_scored
                        + pop_scored
                    )
                    all_candidates = sorted(
                        all_candidates, key=lambda x: x[1], reverse=True
                    )

                    seen, final = set(), []
                    for it, _ in all_candidates:
                        if it not in seen:
                            final.append(it)
                            seen.add(it)
                        if len(final) >= top_k:
                            break

                    # если не хватает, добавляем популярные
                    if len(final) < top_k:
                        for it in popular_items:
                            if it not in seen:
                                final.append(it)
                                seen.add(it)
                            if len(final) >= top_k:
                                break

                    top_items = final[:top_k]

                    batch_rows.append(
                        {
                            "user_id": user_id,
                            "item_id_1 item_id_2 ... item_id_100": " ".join(
                                map(str, top_items)
                            ),
                        }
                    )

                except Exception as e:
                    print(f"Ошибка для пользователя {user_id}: {e}")
                    continue

            # Записываем батч
            if batch_rows:
                pd.DataFrame(batch_rows).to_csv(
                    filename,
                    index=False,
                    mode="w" if first_run else "a",
                    header=first_run,
                )
                first_run = False
                batch_rows = []

            # Очищаем память GPU
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Ошибка в батче пользователей: {e}")
            continue

    if batch_rows:
        pd.DataFrame(batch_rows).to_csv(
            filename, index=False, mode="w" if first_run else "a", header=first_run
        )


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
    catalog_items = items_df.groupby("catalogid")["item_id"].apply(list).to_dict()
    cat_tree = categories_df.set_index("catalogid")["ids"].to_dict()

    similar_map = {}
    for item, cat in items_df[["item_id", "catalogid"]].values:
        sim_items = []
        if cat in cat_tree:
            for parent_cat in cat_tree[cat]:
                sim_items.extend(catalog_items.get(parent_cat, []))
        sim_items = [x for x in sim_items if x != item]
        similar_map[item] = sim_items[:top_n]
    return similar_map


# -------------------- Метрики --------------------
def ndcg_at_k(recommended, ground_truth, k=100):
    if not ground_truth:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 1)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(ground_truth), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ndcg(recommendations, test_df, k=100):
    print("Считаем NDCG...")
    gt = defaultdict(set)
    for r in test_df.itertuples():
        gt[r.user_id].add(r.item_id)
    scores = [
        ndcg_at_k(recs, gt.get(uid, set()), k) for uid, recs in recommendations.items()
    ]
    print("NDCG рассчитан")
    return float(np.mean(scores)) if scores else 0.0


def build_recent_items_map_from_batches(batch_dir, recent_n=5):
    """
    batch_dir: путь к папке с батчами interactions (orders + tracker)
    recent_n: сколько последних товаров хранить для каждого пользователя
    Возвращает словарь user_id -> list последних item_id
    """
    recent_items_map = {}
    batch_files = sorted(Path(batch_dir).glob("*.parquet"))

    for f in batch_files:
        print(f"Обрабатываем батч: {f}")
        df = pd.read_parquet(f, columns=["user_id", "item_id", "timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["user_id", "timestamp"], ascending=[True, False])

        for uid, g in df.groupby("user_id"):
            items = g["item_id"].head(recent_n).tolist()
            if uid not in recent_items_map:
                recent_items_map[uid] = items
            else:
                # объединяем предыдущие и новые, сортируем по времени и оставляем последние recent_n
                combined = pd.Series(recent_items_map[uid] + items)
                recent_items_map[uid] = combined.head(recent_n).tolist()

        del df
        gc.collect()

    return recent_items_map


# -------------------- Сохранение --------------------
def save_model(
    model,
    user_map,
    item_map,
    path="/home/root6/python/e_cup/rec_system/src/models/model_als.pkl",
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "user_map": user_map, "item_map": item_map}, f)
    print(f"Модель сохранена в {path}")


def save_submission_csv(
    recommendations,
    top_k=100,
    filename="/home/root6/python/e_cup/rec_system/result/submission.csv",
    popularity_s=None,
):
    print("Сохраняем submission CSV...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    popular_items = popularity_s.index.tolist() if popularity_s is not None else []

    submission_data = []
    for user_id, items in tqdm(recommendations.items(), desc="Save CSV"):
        items = list(items[:top_k])
        missing = top_k - len(items)
        if missing > 0 and popular_items:
            items.extend([x for x in popular_items if x not in items][:missing])

        submission_data.append(
            {
                "user_id": user_id,
                "item_id_1 item_id_2 ... item_id_100": " ".join(
                    map(str, items[:top_k])
                ),
            }
        )

    pd.DataFrame(submission_data).to_csv(filename, index=False)
    print(f"CSV сохранён: {filename}")


# -------------------- Метрики --------------------
def ndcg_at_k_grouped(predictions, targets, groups, k=100):
    """
    Вычисление NDCG@k для сгруппированных данных (пользователей)
    """
    ndcg_scores = []
    start_idx = 0

    for group_size in groups:
        if group_size == 0:
            continue

        end_idx = start_idx + group_size
        group_preds = predictions[start_idx:end_idx]
        group_targets = targets[start_idx:end_idx]

        # Сортируем предсказания и цели по убыванию score
        sorted_indices = np.argsort(group_preds)[::-1]
        sorted_targets = group_targets[sorted_indices]

        # Вычисляем DCG
        dcg = 0.0
        for i in range(min(k, len(sorted_targets))):
            if sorted_targets[i] > 0:
                dcg += 1.0 / np.log2(i + 2)  # i+2 потому что индекс с 1

        # Вычисляем IDCG
        ideal_sorted = np.sort(group_targets)[::-1]
        idcg = 0.0
        for i in range(min(k, len(ideal_sorted))):
            if ideal_sorted[i] > 0:
                idcg += 1.0 / np.log2(i + 2)

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

        start_idx = end_idx

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


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
                n_negatives = min(3, len(available_negatives))

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
                lgb.early_stopping(50),
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

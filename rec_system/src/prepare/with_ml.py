import gc
import os
import pickle
from collections import defaultdict
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix, csr_matrix, vstack
from tqdm.auto import tqdm

# tqdm интеграция с pandas
tqdm.pandas()


# -------------------- Загрузка данных --------------------
def load_train_data(max_parts=0, max_rows=10000):
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
def build_user_items_csr(batch_files, user_map, item_map, dtype=np.float32):
    """
    batch_files: список parquet файлов с взаимодействиями
    Возвращает CSR матрицу user-items
    """
    from scipy.sparse import coo_matrix, vstack

    print("Строим CSR матрицу по батчам...")
    coo_batches = []

    for f in batch_files:
        df = pd.read_parquet(f)
        df["user_idx"] = df["user_id"].map(user_map).astype(np.int32)
        df["item_idx"] = df["item_id"].map(item_map).astype(np.int32)
        coo = coo_matrix(
            (df["weight"].astype(dtype), (df["user_idx"], df["item_idx"])),
            shape=(len(user_map), len(item_map)),
        )
        coo_batches.append(coo)
        print(f"Батч {f} добавлен")

    user_items_csr = vstack(coo_batches).tocsr()
    print(f"CSR матрица построена: {user_items_csr.shape}")
    return user_items_csr


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
    top_k=100,
    recent_n=5,
    similar_top_n_seed=20,
    blend_sim_beta=0.3,
    blend_cat_beta=0.3,
    batch_size=10_000,
    filename="/home/root6/python/e_cup/rec_system/result/submission.csv",
    item_to_cat=None,
    cat_to_items=None,
):
    import os

    import pandas as pd
    from tqdm import tqdm

    recent_items_map = recent_items_map or {}
    inv_item_map = {int(v): int(k) for k, v in item_map.items()}
    popular_items = popularity_s.index.tolist()

    first_run = True
    if os.path.exists(filename):
        os.remove(filename)

    batch_rows = []

    for user_id in tqdm(users_df["user_id"], desc="Hybrid recommendations iter"):
        try:
            # ======== Загружаем только нужные строки из батчей ========
            user_tracker_list = []
            for f in interactions_files:
                df = pd.read_parquet(f)
                user_rows = df[df["user_id"] == user_id]
                if not user_rows.empty:
                    user_tracker_list.append(user_rows)
            if user_tracker_list:
                user_tracker = pd.concat(user_tracker_list, ignore_index=True)
            else:
                user_tracker = pd.DataFrame(
                    columns=["user_id", "item_id", "weight", "timestamp"]
                )

            user_bought_items = set(user_tracker[user_tracker["weight"] > 0]["item_id"])
            tracker_scored = (
                user_tracker[["item_id", "weight"]].drop_duplicates().values.tolist()
            )

            # ======== ALS рекомендации ========
            als_scored = []
            user_idx_int = user_map.get(user_id)
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
                        category_scored.extend([(i, 0.8) for i in cat_items[:quota]])

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
                + pop_scored
            )
            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)

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

            # записываем батч
            if len(batch_rows) >= batch_size:
                pd.DataFrame(batch_rows).to_csv(
                    filename,
                    index=False,
                    mode="w" if first_run else "a",
                    header=first_run,
                )
                first_run = False
                batch_rows = []

        except Exception as e:
            print(f"Ошибка для пользователя {user_id}: {e}")
            continue

    if batch_rows:
        pd.DataFrame(batch_rows).to_csv(
            filename, index=False, mode="w" if first_run else "a", header=first_run
        )


def evaluate_ndcg_iter(
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

    # создаем ground truth
    gt = test_df.groupby("user_id")["item_id"].apply(set).to_dict()

    ndcg_sum = 0.0
    n_users = 0

    for user_id in tqdm(users_df["user_id"], desc="Evaluate NDCG iter"):
        try:
            user_idx_int = user_map.get(user_id)

            # ======== Загружаем интеракции пользователя из батчей ========
            user_tracker_list = []
            for f in interactions_files:
                df = pd.read_parquet(f)
                user_rows = df[df["user_id"] == user_id]
                if not user_rows.empty:
                    user_tracker_list.append(user_rows)
            if user_tracker_list:
                user_tracker = pd.concat(user_tracker_list, ignore_index=True)
            else:
                user_tracker = pd.DataFrame(
                    columns=["user_id", "item_id", "weight", "timestamp"]
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

            # ======== Объединяем и сортируем ========
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


# -------------------- Основной запуск --------------------
if __name__ == "__main__":
    K = 100
    RECENT_N = 20
    TEST_SIZE = 0.2
    SIMILAR_SEED = 20
    BLEND_SIM_BETA = 0.3

    # -------------------- Load & filter --------------------
    orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_ddf = (
        load_train_data()
    )
    orders_ddf, tracker_ddf, items_ddf = filter_data(orders_ddf, tracker_ddf, items_ddf)

    # -------------------- Split --------------------
    orders_df_full = orders_ddf.compute()
    train_orders_df, test_orders_df, cutoff_ts_per_user = train_test_split_by_time(
        orders_df_full, TEST_SIZE
    )

    # -------------------- Prepare interactions batches --------------------
    interactions_files = prepare_interactions(
        train_orders_df, tracker_ddf, cutoff_ts_per_user, scale_days=30
    )

    # -------------------- Recent items map --------------------
    batch_dir = "/home/root6/python/e_cup/rec_system/data/processed/prepare_interactions_batches"
    recent_items_map = build_recent_items_map_from_batches(batch_dir, recent_n=RECENT_N)

    # -------------------- Train ALS --------------------
    model, user_map, item_map = train_als(interactions_files)
    save_model(
        model,
        user_map,
        item_map,
        path="/home/root6/python/e_cup/rec_system/src/models/model_als.pkl",
    )

    # -------------------- Popularity --------------------
    popularity_s = compute_global_popularity(train_orders_df)

    # -------------------- User-items CSR --------------------
    user_items_csr = build_user_items_csr(interactions_files, user_map, item_map)

    # -------------------- Items & test users --------------------
    with ProgressBar():
        items_df_pd = items_ddf.compute()
    with ProgressBar():
        test_users_df = test_users_ddf.compute()

    # -------------------- Generate submission --------------------
    generate_and_save_recommendations_iter(
        model=model,
        user_map=user_map,
        item_map=item_map,
        user_items_csr=user_items_csr,
        interactions_files=interactions_files,
        popularity_s=popularity_s,
        users_df=test_users_df[["user_id"]],
        recent_items_map=recent_items_map,
        top_k=K,
        similar_top_n_seed=SIMILAR_SEED,
        blend_sim_beta=BLEND_SIM_BETA,
        batch_size=10000,
        filename="/home/root6/python/e_cup/rec_system/result/submission.csv",
    )

    # -------------------- Evaluate NDCG --------------------
    ndcg100 = evaluate_ndcg_iter(
        model=model,
        user_map=user_map,
        item_map=item_map,
        user_items_csr=user_items_csr,
        test_df=test_orders_df[["user_id", "item_id"]],
        popularity_s=popularity_s,
        users_df=test_orders_df[["user_id"]].drop_duplicates(),
        interactions_files=interactions_files,  # <- список батчей
        recent_items_map=recent_items_map,
        top_k=K,
        similar_top_n_seed=SIMILAR_SEED,
        blend_sim_beta=BLEND_SIM_BETA,
    )
    print(f"NDCG@100 (temporal split): {ndcg100:.6f}")
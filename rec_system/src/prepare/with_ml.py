import os
import pickle
from collections import defaultdict

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix, csr_matrix
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
            sample_df = ddf.head(max_rows, compute=True)
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
    Делим покупки каждого пользователя на train/test по времени.
    Возвращаем train_df, test_df и cutoff_ts_per_user (первая дата теста).
    """
    print("Делаем train/test split...")
    orders_df = orders_df.copy()
    orders_df["created_timestamp"] = pd.to_datetime(orders_df["created_timestamp"])

    train_parts, test_parts, cutoff_ts = [], [], {}

    for user, grp in tqdm(orders_df.groupby("user_id"), desc="Split by user"):
        grp = grp.sort_values("created_timestamp")
        n_test = max(1, int(len(grp) * test_size))
        test_g = grp.tail(n_test)
        train_g = grp.iloc[:-n_test] if len(grp) > n_test else grp.iloc[:0]

        if not test_g.empty:
            cutoff_ts[user] = test_g["created_timestamp"].min()

        train_parts.append(train_g)
        test_parts.append(test_g)

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    print("Split завершён")
    return train_df, test_df, cutoff_ts


# -------------------- Подготовка взаимодействий --------------------
def prepare_interactions(
    train_orders_df, tracker_ddf, cutoff_ts_per_user, action_weights=None, scale_days=30
):
    """
    Формируем веса user-item:
    - Заказы (train) с временным весом.
    - Tracker-ивенты до cutoff.
    Сохраняем timestamp для recent_items.
    """
    print("Формируем матрицу взаимодействий...")
    if action_weights is None:
        action_weights = {"page_view": 1, "favorite": 2, "to_cart": 3}

    orders_df = train_orders_df.copy()
    orders_df["created_timestamp"] = pd.to_datetime(orders_df["created_timestamp"])
    ref_time = (
        orders_df["created_timestamp"].max()
        if not orders_df.empty
        else pd.Timestamp.now()
    )
    base_weight = 5.0

    orders_df["days_ago"] = (ref_time - orders_df["created_timestamp"]).dt.days.clip(
        lower=1
    )
    orders_df["time_factor"] = np.log1p(orders_df["days_ago"] / scale_days)
    orders_df["weight"] = base_weight * orders_df["time_factor"]
    orders_df = orders_df.rename(columns={"created_timestamp": "timestamp"})

    tracker_df = tracker_ddf.compute()
    tracker_df["timestamp"] = pd.to_datetime(tracker_df["timestamp"])
    tracker_df["aw"] = tracker_df["action_type"].map(action_weights).fillna(0)
    tracker_df["cutoff"] = tracker_df["user_id"].map(cutoff_ts_per_user)
    tracker_df = tracker_df[
        (tracker_df["cutoff"].isna()) | (tracker_df["timestamp"] < tracker_df["cutoff"])
    ]
    tracker_df["days_ago"] = (ref_time - tracker_df["timestamp"]).dt.days.clip(lower=1)
    tracker_df["time_factor"] = np.log1p(tracker_df["days_ago"] / scale_days)
    tracker_df["weight"] = tracker_df["aw"] * tracker_df["time_factor"]

    interactions_df = pd.concat(
        [
            orders_df[["user_id", "item_id", "weight", "timestamp"]],
            tracker_df[["user_id", "item_id", "weight", "timestamp"]],
        ],
        ignore_index=True,
    )
    interactions_df = interactions_df.groupby(
        ["user_id", "item_id"], as_index=False
    ).agg(
        {"weight": "sum", "timestamp": "max"}  # сохраняем последнее взаимодействие
    )
    print(f"Матрица взаимодействий: {interactions_df.shape}")
    return interactions_df


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
def train_als(interactions_df, factors=64, iterations=100):
    print("Обучаем ALS...")
    user_ids = interactions_df["user_id"].unique()
    item_ids = interactions_df["item_id"].unique()
    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {i: j for j, i in enumerate(item_ids)}

    df = interactions_df.copy()
    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["item_id"].map(item_map)

    sparse_interactions = coo_matrix(
        (
            df["weight"].astype(float),
            (df["user_idx"].astype(int), df["item_idx"].astype(int)),
        ),
        shape=(len(user_ids), len(item_ids)),
    )

    model = AlternatingLeastSquares(
        factors=factors, iterations=iterations, random_state=42
    )

    model.fit(sparse_interactions.T, show_progress=True)

    print("ALS обучение завершено")
    return model, user_map, item_map


# -------------------- User-Items CSR для recommend --------------------
def build_user_items_csr(interactions_df, user_map, item_map):
    print("Строим CSR матрицу user-items...")
    df = interactions_df.copy()
    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["item_id"].map(item_map)
    df = df.dropna(subset=["user_idx", "item_idx"])
    df["user_idx"] = df["user_idx"].astype(int)
    df["item_idx"] = df["item_idx"].astype(int)
    n_users = len(user_map)
    n_items = len(item_map)
    coo = coo_matrix(
        (df["weight"].astype(float), (df["user_idx"], df["item_idx"])),
        shape=(n_users, n_items),
    )
    print("CSR матрица построена")
    return coo.tocsr()


# -------------------- Recommendations (ALS + recent + similar + popular) --------------------
def generate_recommendations_hybrid(
    model,
    user_map,
    item_map,
    user_items_csr,
    interactions_df,
    items_df,
    categories_df,
    users_df,
    popularity_s,
    top_k=100,
    recent_n=5,
    similar_top_n_seed=20,
    blend_sim_beta=0.3,
):
    inv_user_map = {v: k for k, v in user_map.items()}
    inv_item_map = {v: k for k, v in item_map.items()}
    recommendations = {}
    sim_model = model  # пока используем тот же ALS для похожих

    # сортируем interactions для быстрого recent_items
    interactions_df_sorted = interactions_df.sort_values(
        ["user_id", "timestamp"], ascending=[True, False]
    )

    for user_id, user_idx in tqdm(user_map.items(), desc="Hybrid recommendations"):
        try:
            # ALS рекомендации
            rec = model.recommend(
                userid=user_idx,
                user_items=user_items_csr[user_idx],
                N=top_k * 2,
                filter_already_liked_items=True,
                recalculate_user=True,
            )
            als_items = [(inv_item_map[i], float(score)) for i, score in rec]

            # последние товары по timestamp
            recent_items = (
                interactions_df_sorted[interactions_df_sorted["user_id"] == user_id][
                    "item_id"
                ]
                .head(recent_n)
                .tolist()
            )

            # похожие к последним
            sim_items = []
            for rit in recent_items:
                if rit in item_map:
                    sims = sim_model.similar_items(item_map[rit], N=similar_top_n_seed)
                    sim_items.extend([inv_item_map[i] for i, _ in sims])
            sim_items = list(
                dict.fromkeys(sim_items)
            )  # уникальные, порядок сохраняется

            # blend похожих
            sim_quota = int(top_k * blend_sim_beta)
            min_als_score = min((s for _, s in als_items), default=1.0)
            sim_scored = [(it, min_als_score * 0.5) for it in sim_items[:sim_quota]]

            # популярные
            popular_items_full = popularity_s.index.tolist()
            pop_scored = [(it, min_als_score * 0.1) for it in popular_items_full]

            # объединяем и сортируем
            all_candidates = als_items + sim_scored + pop_scored
            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)

            # уникализация top_k
            seen, final = set(), []
            for it, score in all_candidates:
                if it not in seen:
                    final.append(it)
                    seen.add(it)
                if len(final) >= top_k:
                    break

            recommendations[user_id] = final

        except Exception as e:
            print(f"Ошибка для пользователя {user_id}: {e}")
            recommendations[user_id] = popularity_s.index[:top_k].tolist()

    return recommendations


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

        if len(items) < top_k and popular_items:
            for pop_item in popular_items:
                if pop_item not in items:
                    items.append(pop_item)
                if len(items) >= top_k:
                    break

        if len(items) < top_k:
            items.extend([items[0]] * (top_k - len(items)))

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
    RECENT_N = 5
    TEST_SIZE = 0.2
    SIMILAR_SEED = 20
    BLEND_SIM_BETA = 0.3  # доля похожих в топ@K

    # 1) Load & filter
    orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_ddf = (
        load_train_data()
    )
    orders_ddf, tracker_ddf, items_ddf = filter_data(orders_ddf, tracker_ddf, items_ddf)

    # 2) Split by time on PURCHASES
    orders_df_full = orders_ddf.compute()
    train_orders_df, test_orders_df, cutoff_ts_per_user = train_test_split_by_time(
        orders_df_full, test_size=TEST_SIZE
    )

    # 3) Build interactions (train orders + tracker before cutoff)
    interactions_df_train = prepare_interactions(
        train_orders_df, tracker_ddf, cutoff_ts_per_user, scale_days=30
    )

    # 4) Train ALS
    model, user_map, item_map = train_als(interactions_df_train)
    save_model(
        model,
        user_map,
        item_map,
        path="/home/root6/python/e_cup/rec_system/src/models/model_als.pkl",
    )

    # 5) Popularity from TRAIN purchases
    popularity_s = compute_global_popularity(train_orders_df)

    # 6) Матрица user-items CSR для recommend
    user_items_csr = build_user_items_csr(interactions_df_train, user_map, item_map)

    # 7) Recommendations for test users file (submission)
    with ProgressBar():
        items_df_pd = items_ddf.compute()

    with ProgressBar():
        test_users_df = test_users_ddf.compute()

    print("Генерируем рекомендации для SUBMISSION...")
    recs_for_submission = generate_recommendations_hybrid(
        model=model,
        user_map=user_map,
        item_map=item_map,
        user_items_csr=user_items_csr,
        interactions_df=interactions_df_train,
        items_df=items_df_pd,
        categories_df=categories_ddf,
        users_df=test_users_df[["user_id"]],
        popularity_s=popularity_s,
        top_k=K,
        recent_n=RECENT_N,
        similar_top_n_seed=SIMILAR_SEED,
        blend_sim_beta=BLEND_SIM_BETA,
    )
    save_submission_csv(
        recs_for_submission,
        top_k=K,
        filename="/home/root6/python/e_cup/rec_system/result/submission.csv",
    )

    # 8) Evaluation NDCG@100 on temporal test split (users who have test purchases)
    print("Генерируем рекомендации для EVAL...")
    users_with_test_df = test_orders_df[["user_id"]].drop_duplicates()
    recs_for_eval = generate_recommendations_hybrid(
        model=model,
        user_map=user_map,
        item_map=item_map,
        user_items_csr=user_items_csr,
        interactions_df=interactions_df_train,
        items_df=items_df_pd,
        categories_df=categories_ddf,
        users_df=test_users_df[["user_id"]],
        popularity_s=popularity_s,
        top_k=K,
        recent_n=RECENT_N,
        similar_top_n_seed=SIMILAR_SEED,
        blend_sim_beta=BLEND_SIM_BETA,
    )
    ndcg100 = evaluate_ndcg(
        recs_for_eval, test_orders_df[["user_id", "item_id"]], k=100
    )
    print(f"NDCG@100 (temporal split): {ndcg100:.6f}")

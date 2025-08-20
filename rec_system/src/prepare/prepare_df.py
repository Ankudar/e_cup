from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import dask
import dask.dataframe as dd
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from dask.diagnostics import ProgressBar  # type: ignore
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import faiss

device = "cuda" if torch.cuda.is_available() else "cpu"

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


# def load_train_data():
#     print("Загружаем тренировочные данные через Dask...")

#     orders_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_orders_data/*/*.parquet"
#     tracker_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_tracker_data/*/*.parquet"
#     items_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_items_data/*.parquet"
#     categories_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train_final_categories_tree/*.parquet"
#     test_users_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_test_for_participants/test_for_participants/*.parquet"

#     # --- Загружаем заказы ---
#     orders_ddf = dd.read_parquet(orders_path)
#     print(f"Найдено файлов заказов: {orders_ddf.npartitions} частей")

#     # --- Загружаем взаимодействия ---
#     tracker_ddf = dd.read_parquet(tracker_path)
#     print(f"Найдено файлов взаимодействий: {tracker_ddf.npartitions} частей")

#     # --- Загружаем товары ---
#     items_ddf = dd.read_parquet(items_path)
#     print(f"Найдено файлов товаров: {items_ddf.npartitions} частей")

#     # --- Загружаем категории ---
#     categories_ddf = dd.read_parquet(categories_path)
#     print(f"Категорий после фильтрации: {categories_ddf.shape[0].compute():,}")

#     # --- Загружаем тестовых юзеров ---
#     test_users_df = dd.read_parquet(test_users_path)
#     print(f"Тестовых юзеров: {test_users_df.shape[0].compute():,}")

#     # --- Возвращаем Dask DataFrame ---
#     return orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_df


def load_train_data():
    print(
        "Загружаем тренировочные данные через Dask (пробный прогон, максимум 3 партиции на переменную)..."
    )

    orders_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_orders_data/*/*.parquet"
    tracker_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_tracker_data/*/*.parquet"
    items_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_items_data/*.parquet"
    categories_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train_final_categories_tree/*.parquet"
    test_users_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_test_for_participants/test_for_participants/*.parquet"

    def read_sample(path, max_parts=1):
        ddf = dd.read_parquet(path)
        n_parts = min(ddf.npartitions, max_parts)
        return ddf.partitions[:n_parts]

    # --- Загружаем заказы ---
    orders_ddf = read_sample(orders_path)
    print(
        f"Найдено файлов заказов: {orders_ddf.npartitions} частей (ограничено для прогона)"
    )

    # --- Загружаем взаимодействия ---
    tracker_ddf = read_sample(tracker_path)
    print(
        f"Найдено файлов взаимодействий: {tracker_ddf.npartitions} частей (ограничено для прогона)"
    )

    # --- Загружаем товары ---
    items_ddf = read_sample(items_path)
    print(
        f"Найдено файлов товаров: {items_ddf.npartitions} частей (ограничено для прогона)"
    )

    # --- Загружаем категории ---
    categories_ddf = read_sample(categories_path)
    print(
        f"Категорий после фильтрации: {categories_ddf.shape[0].compute():,} (частичный прогон)"
    )

    # --- Загружаем тестовых юзеров ---
    test_users_df = read_sample(test_users_path)
    print(f"Тестовых юзеров: {test_users_df.shape[0].compute():,} (частичный прогон)")

    return orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_df


def filter_items_by_category(items_df, category_df):
    target_ids = {7500, 17777, 7697}  # Одежда, Обувь, Аксессуары
    category_map = category_df[["catalogid", "ids"]].compute()

    # выбираем только те catalogid, где в цепочке есть одна из нужных категорий
    allowed_catalogs = (
        category_map[
            category_map["ids"].apply(lambda x: any(cid in target_ids for cid in x))
        ]["catalogid"]
        .unique()
        .tolist()
    )

    items_filtered = items_df[items_df["catalogid"].isin(allowed_catalogs)]
    print(f"Категорий после фильтрации: {items_filtered.shape[0].compute():,}")

    return items_filtered


def build_user_interactions(orders_df, tracker_df, items_filtered):
    print("=== Start build_user_interactions ===")

    # --- покупки ---
    print("Формируем DataFrame с покупками")
    purchases = orders_df[["user_id", "item_id", "created_timestamp"]].copy()
    purchases = purchases.rename(columns={"created_timestamp": "timestamp"})
    purchases["event_type"] = "purchase"
    print(f"Число строк в purchases: {len(purchases)}")

    # --- просмотры ---
    print("Формируем DataFrame с кликами")
    clicks = tracker_df[["user_id", "item_id", "timestamp"]].copy()
    clicks["event_type"] = "click"
    print(f"Число строк в clicks: {len(clicks)}")

    # объединяем
    print("Объединяем покупки и клики")
    interactions = dd.concat([purchases, clicks])
    print(f"Итого строк в interactions: {interactions.shape[0].compute()}")

    # фильтруем только по оставшимся товарам
    print("Фильтруем только товары из items_filtered")
    items_set = set(items_filtered["item_id"].compute())
    print(f"Количество уникальных item_id для фильтрации: {len(items_set)}")

    # используем map_partitions с фильтром
    def filter_partition(df):
        return df[df["item_id"].isin(items_set)]

    print("Применяем фильтр к каждому чанку с прогрессбаром")
    with ProgressBar():
        interactions_filtered = interactions.map_partitions(filter_partition).persist()

    print("Фильтрация завершена")
    return interactions_filtered


# --- 1) Популярность (быстрый, стабильный генератор) ---
def compute_popularity(interactions_ddf, top_k=1000, window_days=None):
    """
    interactions_ddf: Dask DataFrame с колонками ['item_id','timestamp','event_value']
    event_value — вес (например purchase=3, click=1). Если нет, заменится на 1.
    window_days: если указано, берём только последние window_days дней.
    Возвращает Pandas DataFrame сортированный по убыванию popularity.
    """
    print("=== Start compute_popularity ===")

    if "event_value" not in interactions_ddf.columns:
        print("Добавляем колонку event_value=1 для всех событий")
        interactions_ddf["event_value"] = 1

    print("Приводим timestamp к datetime")
    interactions_ddf["timestamp"] = dd.to_datetime(
        interactions_ddf["timestamp"], errors="coerce"
    )

    if window_days is not None:
        print(f"Фильтруем события за последние {window_days} дней")
        max_ts = interactions_ddf["timestamp"].max().compute()
        cutoff = max_ts - pd.Timedelta(days=window_days)
        interactions_ddf = interactions_ddf[interactions_ddf["timestamp"] >= cutoff]

    print("Считаем сумму event_value по item_id для каждого чанка с прогрессбаром")

    def sum_partition(df):
        return df.groupby("item_id")["event_value"].sum().reset_index()

    with ProgressBar():
        part_sums = interactions_ddf.map_partitions(sum_partition).compute()

    print("Агрегируем результаты всех чанков")
    pop = part_sums.groupby("item_id")["event_value"].sum().reset_index()
    pop = pop.rename(columns={"event_value": "pop_score"})
    pop = pop.sort_values("pop_score", ascending=False).reset_index(drop=True)

    print(f"Возвращаем топ-{top_k} популярных товаров")
    return pop.head(top_k)


# --- 2) Кандидаты из недавней истории пользователя + дозаполнение популярностью ---


def user_recent_candidates_dask(
    interactions_ddf,
    users,
    top_k_per_user=100,
    last_n=30,
    popular_df=None,
    batch_size=20000,
    device="cuda",  # если доступна GPU
):
    """
    Оптимизированная генерация кандидатов последних взаимодействий с Dask и GPU.
    - interactions_ddf: Dask DataFrame ['user_id','item_id','timestamp']
    - users: список user_id
    - last_n: число последних событий для пользователя
    - popular_df: DataFrame с популярными item_id для дозаполнения
    - batch_size: количество пользователей в батче
    - device: 'cuda' или 'cpu'
    """
    print(f"Фильтруем {len(users)} пользователей и берём последние {last_n} событий...")
    interactions_ddf = interactions_ddf[["user_id", "item_id", "timestamp"]].copy()
    interactions_ddf["timestamp"] = dd.to_datetime(
        interactions_ddf["timestamp"], errors="coerce"
    )
    interactions_ddf = interactions_ddf[interactions_ddf["user_id"].isin(users)]

    popular_list = popular_df["item_id"].tolist() if popular_df is not None else None
    result = {}

    # Обработка батчами
    n_batches = (len(users) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch_users = users[b * batch_size : (b + 1) * batch_size]
        batch_ddf = interactions_ddf[interactions_ddf["user_id"].isin(batch_users)]

        # tail последних last_n событий внутри каждой партиции
        def tail_last_n(df):
            return df.sort_values("timestamp").groupby("user_id").tail(last_n)

        with ProgressBar():
            batch_recent = batch_ddf.map_partitions(tail_last_n).compute()

        # Генерация кандидатов
        for uid, user_df in batch_recent.groupby("user_id"):
            his = user_df["item_id"].tolist()
            seen = []
            for it in reversed(his):
                if it not in seen:
                    seen.append(it)
            candidates = list(seen)

            if popular_list is not None:
                idx = 0
                while len(candidates) < top_k_per_user and idx < len(popular_list):
                    if popular_list[idx] not in candidates:
                        candidates.append(popular_list[idx])
                    idx += 1

            result[uid] = candidates[:top_k_per_user]

        print(f"Батч {b+1}/{n_batches} обработан, пользователей: {len(batch_users)}")

    return result


# --- 3) Co-visitation ---
def build_co_visitation(interactions_ddf, max_pairs=5_000_000, window_days=90):
    """
    interactions_ddf: Dask DF с ['user_id','item_id','timestamp','session_id' или 'order_id'].
    Если session_id нет, используем user_id + округлённый timestamp как суррогат.
    max_pairs: ограничение количества пар (для памяти).
    window_days: ограничение по давности.
    Возвращает Pandas DataFrame с ['item_id_x','item_id_y','weight'].
    """
    print("=== Start build_co_visitation ===")
    print("Преобразуем timestamp в datetime...")
    interactions_ddf["timestamp"] = dd.to_datetime(
        interactions_ddf["timestamp"], errors="coerce"
    )

    print(f"Фильтруем взаимодействия за последние {window_days} дней...")
    max_ts = interactions_ddf["timestamp"].max().compute()
    cutoff = max_ts - pd.Timedelta(days=window_days)
    ddf = interactions_ddf[interactions_ddf["timestamp"] >= cutoff]

    # суррогатная сессия
    if "session_id" not in ddf.columns:
        print("Сессии не найдены, создаём суррогатные session_id...")
        ddf["session_id"] = (
            ddf["user_id"].astype(str)
            + "_"
            + ddf["timestamp"].dt.floor("1d").astype(str)
        )

    print("Вычисляем уникальные пары (item_id внутри session)...")
    with ProgressBar():
        sess = ddf[["session_id", "item_id"]].drop_duplicates().compute()

    print(f"Уникальных записей после drop_duplicates: {len(sess):,}")

    # строим пары
    print("Формируем пары item-item внутри сессий...")
    pairs = sess.merge(sess, on="session_id")
    pairs = pairs[pairs["item_id_x"] != pairs["item_id_y"]]
    print(f"Количество пар после фильтрации self-pairs: {len(pairs):,}")

    # считаем веса
    print("Группируем пары и считаем веса...")
    co = pairs.groupby(["item_id_x", "item_id_y"]).size().reset_index(name="weight")
    co = co.sort_values("weight", ascending=False)

    if max_pairs:
        print(f"Ограничиваем количество пар до {max_pairs:,}...")
        co = co.head(max_pairs)

    print(
        f"=== build_co_visitation завершено, итоговое количество пар: {len(co):,} ==="
    )
    return co


def user_covis_candidates(
    interactions_ddf,
    users,
    co_matrix,
    top_k_per_user=100,
    popular_df=None,
    batch_size=500,
    n_jobs=8,
):
    print("=== Start user_covis_candidates_fast ===")
    with ProgressBar():
        inter = interactions_ddf[interactions_ddf["user_id"].isin(users)].compute()
    inter = inter.sort_values("timestamp")

    # создаём историю пользователей
    print("Создаём словарь user_history...")
    user_history = {uid: df["item_id"].tolist() for uid, df in inter.groupby("user_id")}

    # создаём co_dict
    print("Создаём словарь co_dict...")
    co_dict = defaultdict(list)
    for row in tqdm(co_matrix.itertuples(index=False), desc="Building co_dict"):
        co_dict[row.item_id_x].append((row.item_id_y, row.weight))

    popular_list = popular_df["item_id"].tolist() if popular_df is not None else None

    def process_batch(batch_users):
        batch_result = {}
        for uid in batch_users:
            his = user_history.get(uid, [])
            cands = []
            for it in his:
                neigh = sorted(co_dict.get(it, []), key=lambda x: -x[1])
                for nid, _ in neigh:
                    if nid not in his and nid not in cands:
                        cands.append(nid)
                    if len(cands) >= top_k_per_user:
                        break
                if len(cands) >= top_k_per_user:
                    break
            # дозаполнение популярными
            if popular_list is not None and len(cands) < top_k_per_user:
                for pid in popular_list:
                    if pid not in his and pid not in cands:
                        cands.append(pid)
                    if len(cands) >= top_k_per_user:
                        break
            batch_result[uid] = cands[:top_k_per_user]
        return batch_result

    result = {}
    batches = [users[i : i + batch_size] for i in range(0, len(users), batch_size)]
    with ThreadPoolExecutor(n_jobs) as executor:
        for batch_res in tqdm(
            executor.map(process_batch, batches),
            total=len(batches),
            desc="Co-visitation batching",
        ):
            result.update(batch_res)

    print(f"=== Finished, users: {len(result):,} ===")
    return result


# --- 4) ANN по fclip_embed ---
def ann_candidates(
    user_history_ids,
    item_embeddings,
    ids,
    embs_tensor,
    top_k=100,
    device="cpu",
    batch_size=50,
):
    """
    user_history_ids: список item_id пользователя
    item_embeddings: DataFrame с эмбеддингами (для выборки query)
    ids: np.array всех item_id
    embs_tensor: torch.Tensor всех эмбеддингов
    """
    print("=== Start ANN candidates ===")
    print(f"Количество товаров в истории пользователя: {len(user_history_ids):,}")

    candidates = []

    # Разбиваем на батчи, чтобы не перегружать память
    for i in tqdm(
        range(0, len(user_history_ids), batch_size), desc="Processing ANN batches"
    ):
        batch_ids = user_history_ids[i : i + batch_size]

        queries = []
        for item_id in batch_ids:
            emb_vals = item_embeddings.loc[
                item_embeddings["item_id"] == item_id, "fclip_embed"
            ].values
            if len(emb_vals) == 0:
                continue
            queries.append(
                torch.tensor(emb_vals[0], dtype=torch.float32, device=device)
            )

        if not queries:
            continue

        queries_tensor = torch.stack(queries)
        queries_tensor = torch.nn.functional.normalize(queries_tensor, dim=1)

        # косинусная близость через dot-product
        scores = queries_tensor @ embs_tensor.T
        topk_indices = torch.topk(scores, k=top_k, dim=1).indices

        for inds in topk_indices:
            for idx in inds:
                iid = ids[idx]
                if iid not in user_history_ids and iid not in candidates:
                    candidates.append(iid)
                if len(candidates) >= top_k:
                    break
            if len(candidates) >= top_k:
                break

        if len(candidates) >= top_k:
            break

    print(f"=== ANN candidates done, total candidates: {len(candidates):,} ===")
    return candidates[:top_k]


def merge_candidates(
    users, pop_cands, recent_cands, covis_cands, ann_cands, top_k=100, popular_df=None
):
    """
    users: список user_id
    *_cands: dict[user_id] -> list[item_id]
    popular_df: DataFrame ['item_id'] для дозаполнения
    """
    final = {}
    for uid in users:
        seen = set()
        merged = []

        for src in [pop_cands, recent_cands, covis_cands, ann_cands]:
            if src is None or uid not in src:
                continue
            for it in src[uid]:
                if it not in seen:
                    merged.append(it)
                    seen.add(it)
                if len(merged) >= top_k:
                    break
            if len(merged) >= top_k:
                break

        # дозаполнение популярными
        if popular_df is not None and len(merged) < top_k:
            for pid in popular_df["item_id"]:
                if pid not in seen:
                    merged.append(pid)
                    seen.add(pid)
                if len(merged) >= top_k:
                    break

        final[uid] = merged[:top_k]

    return final


def build_train_set(candidates, interactions, val_data):
    """
    candidates: dict[user_id -> list[item_id]]
    interactions: train взаимодействия
    val_data: покупки из holdout периода
    """
    rows = []
    # создаем set покупок валидации для быстрого поиска
    val_purchases = set(zip(val_data["user_id"], val_data["item_id"]))

    for uid, items in candidates.items():
        for iid in items:
            label = 1 if (uid, iid) in val_purchases else 0
            rows.append((uid, iid, label))

    df = pd.DataFrame(rows, columns=["user_id", "item_id", "label"])
    return df


def add_features(df, orders_df, items_df, cat_pop, covis_df):
    """
    df: user-item-label
    orders_df: все заказы (train)
    items_df: справочник товаров (категории, бренд и т.д.)
    cat_pop: популярность категорий
    covis_df: матрица ковизитов (item-item связи)
    """
    # 1. Популярность товара
    item_pop = orders_df.groupby("item_id")["user_id"].count().rename("item_pop")
    df = df.merge(item_pop, on="item_id", how="left").fillna(0)

    # 2. Популярность категории
    df = df.merge(items_df[["item_id", "category_id"]], on="item_id", how="left")
    df = df.merge(cat_pop.rename("cat_pop"), on="category_id", how="left").fillna(0)

    # 3. Активность пользователя
    user_act = orders_df.groupby("user_id")["item_id"].count().rename("user_activity")
    df = df.merge(user_act, on="user_id", how="left").fillna(0)

    # 4. Ковизитные фичи (например, max co-vis count между candidate item и item'ами юзера)
    def covis_score(uid, iid):
        user_items = orders_df.loc[orders_df["user_id"] == uid, "item_id"].unique()
        scores = [covis_df.get((iid, x), 0) for x in user_items]
        return max(scores) if scores else 0

    df["covis_max"] = df.apply(lambda x: covis_score(x.user_id, x.item_id), axis=1)

    return df


def train_ranker(df):
    features = [c for c in df.columns if c not in ["user_id", "item_id", "label"]]
    X = df[features]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=500,
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=50)],
    )

    y_pred = np.array(model.predict(X_val, num_iteration=model.best_iteration))
    auc = roc_auc_score(y_val, y_pred)
    print("Validation AUC:", auc)

    return model, features


def recommend_topk(model, features, cand_df, K=100):
    """
    model: обученный LightGBM
    features: список фич
    cand_df: dataframe user_id | item_id + фичи
    """
    X = cand_df[features]
    cand_df["score"] = model.predict(X, num_iteration=model.best_iteration)

    # сортируем внутри каждого пользователя
    topk = (
        cand_df.sort_values(["user_id", "score"], ascending=[True, False])
        .groupby("user_id")
        .head(K)
        .reset_index(drop=True)
    )
    return topk[["user_id", "item_id", "score"]]


def make_submission(topk, K=100, path="submission.csv"):
    # topk: user_id | item_id | score
    # берём top-K по каждому пользователю
    topk100 = (
        topk.sort_values(["user_id", "score"], ascending=[True, False])
        .groupby("user_id")["item_id"]
        .apply(lambda x: list(x)[:K])
        .reset_index()
    )

    # формируем колонку с товарами через пробел
    topk100["items"] = topk100["item_id"].apply(lambda ids: " ".join(map(str, ids)))

    # финальный формат
    submission = topk100[["user_id", "items"]].rename(
        columns={"items": "item_id_1 item_id_2 ... item_id_100"}
    )

    submission.to_csv(path, index=False)
    return submission


# --- 4) ANN по fclip_embed (PyTorch CPU/на GPU при наличии) ---
def build_ann_index(item_embeddings, emb_col="fclip_embed", device="cpu"):
    """
    item_embeddings: DataFrame ['item_id','fclip_embed'] где fclip_embed = np.array
    Возвращает:
        ids: np.array item_id
        embs_tensor: torch.Tensor нормализованные эмбеддинги
    """
    ids = item_embeddings["item_id"].values
    embs = np.stack(item_embeddings[emb_col].values).astype("float32")
    embs_tensor = torch.tensor(embs, dtype=torch.float32, device=device)
    embs_tensor = torch.nn.functional.normalize(embs_tensor, dim=1)
    return ids, embs_tensor


if __name__ == "__main__":
    # --- 1) Загрузка данных через Dask ---
    print("=== 1) Загрузка данных через Dask ===")
    with ProgressBar():
        orders_ddf, tracker_ddf, items_ddf, category_ddf, test_users_ddf = (
            load_train_data()
        )
    print("Данные загружены.")
    print(items_ddf.head(2))
    print(orders_ddf.head(2))

    # --- 2) Фильтрация товаров по категориям ---
    print("=== 2) Фильтрация товаров по категориям ===")
    items_filtered_ddf = filter_items_by_category(items_ddf, category_ddf)
    print(f"Оставлено товаров: {len(items_filtered_ddf)}")

    # --- 3) Взаимодействия пользователей ---
    print("=== 3) Построение взаимодействий пользователей ===")
    interactions_ddf = build_user_interactions(
        orders_ddf, tracker_ddf, items_filtered_ddf
    )
    print("Взаимодействия построены.")

    # --- 4) Популярные товары (топ 2000) ---
    print("=== 4) Вычисление популярных товаров ===")
    top_pop_df = compute_popularity(interactions_ddf, top_k=2000, window_days=90)
    print("Популярные товары готовы.")

    # --- 5) Кандидаты из последних действий пользователя ---
    print("=== 5) Формируем кандидатов из последних действий пользователя ===")
    test_users_list = test_users_ddf["user_id"].compute().tolist()
    recent_cands = user_recent_candidates_dask(
        interactions_ddf,
        test_users_list,
        top_k_per_user=100,
        last_n=30,
        popular_df=top_pop_df,
    )
    print("Кандидаты из последних действий сформированы.")

    # --- 6) Co-visitation ---
    print("=== 6) Co-visitation ===")
    co_mat_df = build_co_visitation(
        interactions_ddf, max_pairs=2_000_000, window_days=60
    )
    covis_cands = user_covis_candidates(
        interactions_ddf,
        test_users_list,
        co_mat_df,
        top_k_per_user=100,
        popular_df=top_pop_df,
    )
    print("Co-visitation кандидаты готовы.")

    # --- 7) ANN по fclip_embed через Faiss ---
    print("=== 7) ANN через Faiss ===")
    item_embeddings_df = items_filtered_ddf[["item_id", "fclip_embed"]].compute()
    ids = item_embeddings_df["item_id"].values
    embs_np = np.stack(item_embeddings_df["fclip_embed"].values).astype(np.float32)

    print("Нормализация эмбеддингов...")
    faiss.normalize_L2(embs_np)

    print("Создание Faiss индекса...")
    d = embs_np.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs_np)

    embeddings_map = {
        iid: emb for iid, emb in zip(item_embeddings_df["item_id"], embs_np)
    }

    # --- Формирование историй пользователей с прогрессбаром ---
    print("Формируем истории пользователей...")
    interactions_df = interactions_ddf.compute().sort_values("timestamp")
    user_histories = {}
    for uid in tqdm(test_users_list, desc="User histories"):
        user_histories[uid] = interactions_df[interactions_df["user_id"] == uid][
            "item_id"
        ].tolist()
    print(f"Истории сформированы для {len(user_histories)} пользователей.")

    # --- ANN рекомендации с прогрессбаром (батчи пользователей) ---
    print("Поиск ANN кандидатов (батчи)...")
    ann_cands = {}
    batch_size = 500
    uids = list(user_histories.keys())
    use_last_n = 5  # количество последних элементов для использования. 0 - если надо всё использовать

    for i in tqdm(range(0, len(uids), batch_size), desc="ANN via Faiss (batches)"):
        batch_uids = uids[i : i + batch_size]
        batch_query_embs = []
        batch_map = []

        # формируем один большой массив запросов для батча
        for uid in batch_uids:
            if use_last_n:  # Если используем последние N элементов
                his = user_histories[uid][-use_last_n:]  # последние N
            else:
                his = user_histories[uid]  # все элементы

            for iid in his:
                if iid in embeddings_map:
                    batch_query_embs.append(embeddings_map[iid])
                    batch_map.append(uid)

        if not batch_query_embs:
            continue

        batch_query_embs = np.stack(batch_query_embs)
        faiss.normalize_L2(batch_query_embs)

        # поиск в Faiss
        D, I = index.search(batch_query_embs, k=100)

        # разрезаем результаты по пользователям
        for uid, inds in zip(batch_map, I):
            if uid not in ann_cands:
                ann_cands[uid] = []
            for ix in inds:
                iid = ids[ix]
                if iid not in user_histories[uid] and iid not in ann_cands[uid]:
                    ann_cands[uid].append(iid)
                if len(ann_cands[uid]) >= 100:
                    break

    print(f"ANN кандидаты сформированы для {len(ann_cands):,} пользователей")

    # --- 8) Объединяем кандидатов ---
    print("=== 8) Объединяем кандидатов из всех источников ===")
    final_candidates = merge_candidates(
        users=test_users_list,
        pop_cands={uid: top_pop_df["item_id"].tolist() for uid in test_users_list},
        recent_cands=recent_cands,
        covis_cands=covis_cands,
        ann_cands=ann_cands,
        top_k=100,
        popular_df=top_pop_df,
    )
    print("Кандидаты объединены.")

    # --- 9) Формируем train set для ранжировщика ---
    print("=== 9) Формируем train set ===")
    val_cutoff = interactions_df["timestamp"].quantile(0.9)
    val_data = interactions_df[interactions_df["timestamp"] >= val_cutoff]
    train_df = build_train_set(final_candidates, interactions_df, val_data)
    print(f"Размер train set: {train_df.shape}")

    # --- 10) Добавляем фичи ---
    print("=== 10) Добавляем фичи ===")

    # Вычисляем Pandas DataFrame из Dask
    orders_df = orders_ddf.compute()
    items_df = items_filtered_ddf.compute()

    # Проверяем наличие category_id, если нет — добавляем из category_ddf
    # if "category_id" not in items_df.columns:
    #     print("category_id отсутствует в items_df, добавляем из category_ddf...")
    #     items_df = items_df.merge(
    #         category_ddf[["item_id", "category_id"]], on="item_id", how="left"
    #     )

    # Вычисляем популярность категорий
    cat_pop = (
        orders_df.merge(items_df[["item_id", "catalogid"]], on="item_id", how="left")
        .groupby("catalogid")["user_id"]
        .count()
    )
    print(
        f"Популярность категорий рассчитана, уникальных категорий: {cat_pop.shape[0]}"
    )

    # Создаем словарь co-visitation
    covis_dict = {
        (row.item_id_x, row.item_id_y): row.weight
        for row in co_mat_df.itertuples(index=False)
    }
    print(f"Словарь co-visitation содержит {len(covis_dict)} пар товаров")

    # Добавляем фичи к train_df
    train_df = add_features(train_df, interactions_df, items_df, cat_pop, covis_dict)
    print("Фичи добавлены.")

    # --- 11) Обучаем ранжировщик ---
    print("=== 11) Обучаем модель ранжировщика ===")
    model, features = train_ranker(train_df)
    print("Ранжировщик обучен.")

    # --- 12) Получаем top-K рекомендации по ранжировщику ---
    print("=== 12) Формируем top-K рекомендации по ранжировщику ===")
    topk_df = recommend_topk(model, features, train_df, K=100)
    print("Top-K рекомендации готовы.")

    # --- 13) Submission ---
    print("=== 13) Создаём файл submission ===")
    submission = make_submission(topk_df, K=100, path="submission.csv")
    print("Submission готова:")
    print(submission.head())

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
from sklearn.metrics import ndcg_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm.auto import tqdm

import faiss

tqdm.pandas()

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


def load_train_data(max_parts=1, max_rows=500):
    """
    Загружает данные через Dask с опцией частичного прогона.
    max_parts: сколько частей Dask загружать (для orders, tracker, items)
    max_rows: ограничение по числу строк (для orders, tracker, items)
    categories и test_users всегда берутся полностью
    """
    print("=== Загрузка данных через Dask (пробный прогон) ===")

    paths = {
        "orders": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_orders_data/*/*.parquet",
        "tracker": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_tracker_data/*/*.parquet",
        "items": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_items_data/*.parquet",
        "categories": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train_final_categories_tree/*.parquet",
        "test_users": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_test_for_participants/test_for_participants/*.parquet",
    }

    def read_sample(path, max_parts=max_parts, max_rows=max_rows):
        """Частичная загрузка (ограниченные данные)"""
        ddf = dd.read_parquet(path)
        n_parts = min(ddf.npartitions, max_parts)
        ddf = ddf.partitions[:n_parts]
        if max_rows is not None:
            sample_df = ddf.head(max_rows, compute=True)
            ddf = dd.from_pandas(sample_df, npartitions=1)
        return ddf

    def read_full(path):
        """Полная загрузка (без ограничений)"""
        return dd.read_parquet(path)

    # Ограниченные данные
    orders_ddf = read_sample(paths["orders"])
    print(
        f"Заказы: {orders_ddf.shape[0].compute():,} строк ({orders_ddf.npartitions} частей)"
    )

    tracker_ddf = read_sample(paths["tracker"])
    print(
        f"Взаимодействия: {tracker_ddf.shape[0].compute():,} строк ({tracker_ddf.npartitions} частей)"
    )

    items_ddf = read_sample(paths["items"])
    print(
        f"Товары: {items_ddf.shape[0].compute():,} строк ({items_ddf.npartitions} частей)"
    )

    # Полные данные
    categories_ddf = read_full(paths["categories"])
    print(f"Категории: {categories_ddf.shape[0].compute():,} строк (полный прогон)")

    test_users_ddf = read_full(paths["test_users"])
    print(
        f"Тестовые пользователи: {test_users_ddf.shape[0].compute():,} строк (полный прогон)"
    )

    return orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_ddf


def filter_items_by_category(items_ddf, category_ddf):
    """
    Фильтрует товары по заданным категориям.
    target_ids: категории, которые хотим оставить
    """
    target_ids = {7500, 17777, 7697}  # Одежда, Обувь, Аксессуары

    # Берем только нужные столбцы
    category_map = category_ddf[["catalogid", "ids"]]

    # Создаем флаг: есть ли нужная категория в списке ids
    def has_target(x):
        return bool(set(x) & target_ids)

    category_map = category_map.assign(
        flag=category_map["ids"].map(has_target, meta=("flag", "bool"))
    )

    # Оставляем только разрешенные catalogid
    allowed_catalogs = (
        category_map[category_map["flag"]]["catalogid"].compute().tolist()
    )

    # Фильтруем товары
    items_filtered = items_ddf[items_ddf["catalogid"].isin(allowed_catalogs)]
    print(f"Товаров после фильтрации: {items_filtered.shape[0].compute():,}")

    return items_filtered


def build_user_interactions(orders_ddf, tracker_ddf, items_filtered_ddf):
    print("=== Start build_user_interactions ===")

    # --- покупки ---
    print("Формируем DataFrame с покупками")
    purchases = orders_ddf[["user_id", "item_id", "created_timestamp"]].rename(
        columns={"created_timestamp": "timestamp"}
    )
    purchases = purchases.assign(event_type="purchase")
    print(
        f"Число строк в purchases: {len(purchases)}"
    )  # len на Dask вернет np.nan, можно убрать

    # --- просмотры ---
    print("Формируем DataFrame с кликами")
    clicks = tracker_ddf[["user_id", "item_id", "timestamp"]].assign(event_type="click")
    print(f"Число строк в clicks: {len(clicks)}")

    # объединяем
    print("Объединяем покупки и клики")
    interactions = dd.concat([purchases, clicks])
    print(f"Итого строк в interactions: {interactions.shape[0].compute():,}")

    # фильтруем только по оставшимся товарам
    print("Фильтруем только товары из items_filtered")
    items_set = set(items_filtered_ddf["item_id"].compute())
    print(f"Количество уникальных item_id для фильтрации: {len(items_set)}")

    # фильтрация через map_partitions
    def filter_partition(df):
        return df[df["item_id"].isin(items_set)]

    print("Применяем фильтр к каждому чанку с прогрессбаром")
    with ProgressBar():
        interactions_filtered = interactions.map_partitions(filter_partition).persist()

    print(
        f"Фильтрация завершена, всего строк: {interactions_filtered.shape[0].compute():,}"
    )
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

    print("Считаем сумму event_value по item_id с прогрессбаром")
    with ProgressBar():
        pop = interactions_ddf.groupby("item_id")["event_value"].sum().compute()

    pop = pop.reset_index().rename(columns={"event_value": "pop_score"})
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
    device="cuda",
):
    """
    Генерация кандидатов последних взаимодействий с Dask.
    - interactions_ddf: Dask DataFrame ['user_id','item_id','timestamp']
    - users: список user_id
    - last_n: число последних событий для пользователя
    - popular_df: DataFrame с популярными item_id для дозаполнения
    - batch_size: количество пользователей в батче
    - device: 'cuda' или 'cpu' (для совместимости, сейчас не используется)
    """
    print(f"Фильтруем {len(users)} пользователей и берём последние {last_n} событий...")

    interactions_ddf = interactions_ddf[["user_id", "item_id", "timestamp"]].copy()
    interactions_ddf["timestamp"] = dd.to_datetime(
        interactions_ddf["timestamp"], errors="coerce"
    )
    interactions_ddf = interactions_ddf[interactions_ddf["user_id"].isin(users)]

    popular_list = popular_df["item_id"].tolist() if popular_df is not None else []
    result = {}

    n_batches = (len(users) + batch_size - 1) // batch_size

    for b in range(n_batches):
        batch_users = users[b * batch_size : (b + 1) * batch_size]
        batch_ddf = interactions_ddf[interactions_ddf["user_id"].isin(batch_users)]

        # tail последних last_n событий внутри каждой партиции
        def tail_last_n(df):
            return (
                df.sort_values("timestamp").groupby("user_id", sort=False).tail(last_n)
            )

        with ProgressBar():
            batch_recent = batch_ddf.map_partitions(tail_last_n).compute()

        # Формируем уникальный список последних interacted items + дозаполнение популярностью
        for uid, user_df in batch_recent.groupby("user_id"):
            his = user_df["item_id"].tolist()
            seen = []
            for it in reversed(his):
                if it not in seen:
                    seen.append(it)
            candidates = seen

            # Дополняем популярными товарами, если нужно
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
    Генерация co-visitation матрицы.
    interactions_ddf: Dask DF с ['user_id','item_id','timestamp','session_id' (опционально)]
    max_pairs: ограничение по числу пар
    window_days: ограничение по давности событий
    """
    print("=== Start build_co_visitation ===")
    interactions_ddf["timestamp"] = dd.to_datetime(
        interactions_ddf["timestamp"], errors="coerce"
    )

    # фильтр по времени
    print(f"Фильтруем события за последние {window_days} дней...")
    max_ts = interactions_ddf["timestamp"].max().compute()
    cutoff = max_ts - pd.Timedelta(days=window_days)
    ddf = interactions_ddf[interactions_ddf["timestamp"] >= cutoff]

    # создаём session_id, если нет
    if "session_id" not in ddf.columns:
        print("Сессии не найдены, создаём суррогатные session_id...")
        ddf["session_id"] = (
            ddf["user_id"].astype(str)
            + "_"
            + ddf["timestamp"].dt.floor("1d").astype(str)
        )

    # уникальные item_id внутри сессий
    print("Вычисляем уникальные пары внутри сессий...")
    with ProgressBar():
        sess = ddf[["session_id", "item_id"]].drop_duplicates().compute()
    print(f"Уникальных записей после drop_duplicates: {len(sess):,}")

    # формируем item-item пары
    print("Формируем пары item-item внутри сессий...")
    pairs = sess.merge(sess, on="session_id")
    pairs = pairs[pairs["item_id_x"] != pairs["item_id_y"]]
    print(f"Количество пар после фильтрации self-pairs: {len(pairs):,}")

    # считаем вес каждой пары (сколько раз встретились вместе)
    print("Группируем пары и считаем веса...")
    co = pairs.groupby(["item_id_x", "item_id_y"]).size().reset_index(name="weight")
    co = co.sort_values("weight", ascending=False)

    # ограничение по max_pairs
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
    print("=== Start user_covis_candidates ===")

    # фильтруем пользователей и приводим к Pandas
    with ProgressBar():
        inter = interactions_ddf[interactions_ddf["user_id"].isin(users)].compute()
    inter = inter.sort_values("timestamp")

    # словарь истории пользователей
    print("Создаём user_history...")
    user_history = {uid: df["item_id"].tolist() for uid, df in inter.groupby("user_id")}

    # словарь co-visitation
    print("Создаём co_dict...")
    co_dict = defaultdict(list)
    for row in tqdm(co_matrix.itertuples(index=False), desc="Building co_dict"):
        co_dict[row.item_id_x].append((row.item_id_y, row.weight))

    popular_list = popular_df["item_id"].tolist() if popular_df is not None else None

    # обработка одного батча пользователей
    def process_batch(batch_users):
        batch_result = {}
        for uid in batch_users:
            his = set(user_history.get(uid, []))
            cands = []
            for it in user_history.get(uid, []):
                neighbors = sorted(co_dict.get(it, []), key=lambda x: -x[1])
                for nid, _ in neighbors:
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

    # разбиваем пользователей на батчи
    batches = [users[i : i + batch_size] for i in range(0, len(users), batch_size)]
    result = {}
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
    item_embeddings: DataFrame с колонкой 'fclip_embed' для каждого item_id
    ids: np.array всех item_id, соответствующий embs_tensor
    embs_tensor: torch.Tensor всех эмбеддингов
    """
    print("=== Start ANN candidates ===")
    print(f"Количество товаров в истории пользователя: {len(user_history_ids):,}")

    candidates = []
    seen_set = set(user_history_ids)  # ускоряем проверку уникальности

    for i in tqdm(
        range(0, len(user_history_ids), batch_size), desc="Processing ANN batches"
    ):
        batch_ids = user_history_ids[i : i + batch_size]
        queries = []

        # собираем эмбеддинги батча
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

        # добавляем кандидатов по порядку
        for inds in topk_indices:
            for idx in inds:
                iid = ids[idx]
                if iid not in seen_set:
                    candidates.append(iid)
                    seen_set.add(iid)
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
    Объединяет кандидатов из разных источников и дозаполняет популярными.

    users: список user_id
    *_cands: dict[user_id] -> list[item_id]
    popular_df: DataFrame ['item_id'] для дозаполнения
    """
    final = {}
    popular_list = popular_df["item_id"].tolist() if popular_df is not None else []

    for uid in users:
        seen = set()
        merged = []

        for src in [pop_cands, recent_cands, covis_cands, ann_cands]:
            if not src or uid not in src:
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
        if len(merged) < top_k:
            for pid in popular_list:
                if pid not in seen:
                    merged.append(pid)
                    seen.add(pid)
                if len(merged) >= top_k:
                    break

        final[uid] = merged[:top_k]

    return final


def build_train_set(candidates, interactions, val_data):
    """
    Формирует обучающую выборку для модели.

    candidates: dict[user_id -> list[item_id]]
    interactions: Dask/Pandas DataFrame с train взаимодействиями (не используется здесь)
    val_data: Pandas DataFrame с покупками из holdout периода
    """
    print("=== Start build_train_set ===")
    # создаем set покупок валидации для быстрого поиска
    val_purchases = set(zip(val_data["user_id"], val_data["item_id"]))

    data = [
        (uid, iid, int((uid, iid) in val_purchases))
        for uid, items in candidates.items()
        for iid in items
    ]

    df = pd.DataFrame(data, columns=["user_id", "item_id", "label"])
    print(f"Train set сформирован: {len(df):,} строк")
    return df


def add_features(df, orders_df, items_df, cat_pop, covis_dict):
    """
    df: user-item-label
    orders_df: все заказы (train), содержит ['user_id','item_id','created_timestamp','catalogid','price' optional]
    items_df: справочник товаров (категории, бренд и т.д.)
    cat_pop: популярность категорий (series: index=catalogid)
    covis_dict: словарь {(item1, item2): score}
    """
    print("=== Start add_features ===")

    # --- 1. Популярность товара ---
    print("1. Популярность товара")
    item_pop = orders_df.groupby("item_id")["user_id"].count().rename("item_pop")
    df = df.merge(item_pop, on="item_id", how="left").fillna(0)

    # --- 2. Популярность категории ---
    print("2. Популярность категории")
    df = df.merge(items_df[["item_id", "catalogid"]], on="item_id", how="left")
    df = df.merge(cat_pop.rename("cat_pop"), on="catalogid", how="left").fillna(0)

    # --- 3. Активность пользователя ---
    print("3. Активность пользователя")
    user_act = orders_df.groupby("user_id")["item_id"].count().rename("user_activity")
    df = df.merge(user_act, on="user_id", how="left").fillna(0)

    # --- 4. Поведенческие признаки пользователя ---
    print("4. Поведенческие признаки пользователя")
    user_unique_items = (
        orders_df.groupby("user_id")["item_id"].nunique().rename("user_unique_items")
    )
    df = df.merge(user_unique_items, on="user_id", how="left").fillna(0)

    orders_df["created_timestamp"] = pd.to_datetime(orders_df["created_timestamp"])
    user_last_ts = (
        orders_df.groupby("user_id")["created_timestamp"].max().rename("user_last_ts")
    )
    df = df.merge(
        (pd.Timestamp("today") - user_last_ts).dt.days.rename("user_recency"),
        on="user_id",
        how="left",
    ).fillna(999)

    # --- 5. Ковизитные фичи ---
    print("5. Ковизитные фичи")
    user_items_map = orders_df.groupby("user_id")["item_id"].agg(set).to_dict()

    def calc_covis_features(user_ids, item_ids):
        covis_max = np.zeros(len(user_ids), dtype=np.float32)
        covis_sum = np.zeros(len(user_ids), dtype=np.float32)
        covis_count = np.zeros(len(user_ids), dtype=np.float32)

        for idx in tqdm(
            range(len(user_ids)), desc="Calculating co-visitation features"
        ):
            uid = user_ids[idx]
            iid = item_ids[idx]
            user_items = user_items_map.get(uid, set())
            if user_items:
                scores = [covis_dict.get((iid, x), 0) for x in user_items]
                covis_max[idx] = max(scores)
                covis_sum[idx] = sum(scores)
                covis_count[idx] = sum([1 for s in scores if s > 0])
        return covis_max, covis_sum, covis_count

    covis_max, covis_sum, covis_count = calc_covis_features(
        df["user_id"].values, df["item_id"].values
    )
    df["covis_max"] = covis_max
    df["covis_sum"] = covis_sum
    df["covis_count"] = covis_count

    # --- 6. User-Item взаимодействие ---
    print("6. User-Item взаимодействия")
    user_item_count = (
        orders_df.groupby(["user_id", "item_id"])["created_timestamp"]
        .count()
        .rename("user_item_count")
    )
    df = df.merge(user_item_count, on=["user_id", "item_id"], how="left").fillna(0)

    last_purchase = orders_df.groupby(["user_id", "item_id"])["created_timestamp"].max()
    df = df.merge(
        (pd.Timestamp("today") - last_purchase).dt.days.rename("user_item_days_ago"),
        on=["user_id", "item_id"],
        how="left",
    ).fillna(999)

    print("=== add_features finished ===")
    return df


def train_ranker(train_df, val_df=None):
    """
    train_df: DataFrame с колонками ['user_id', 'item_id', 'label', ...features]
    val_df: DataFrame для валидации, если None - обучение на всей выборке без early stopping
    """
    features = [c for c in train_df.columns if c not in ["user_id", "item_id", "label"]]

    X_train = train_df[features]
    y_train = train_df["label"]
    group_train = train_df.groupby("user_id").size().to_numpy()

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_at=[100],
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=20,  # минимальное число строк на лист
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        verbose=-1,  # отключаем предупреждения
    )

    if val_df is not None and len(val_df) > 0:
        X_val = val_df[features]
        y_val = val_df["label"]
        group_val = val_df.groupby("user_id").size().to_numpy()

        model.fit(
            X_train,
            y_train,
            group=group_train,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val],
            eval_at=[100],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(50),
            ],
        )
    else:
        model.fit(X_train, y_train, group=group_train)

    return model, features


# --- 4) ANN по fclip_embed (PyTorch CPU/GPU) ---
def build_ann_index(item_embeddings, emb_col="fclip_embed", device="cpu"):
    if item_embeddings.empty:
        raise ValueError("item_embeddings пустой")

    ids = item_embeddings["item_id"].values

    def parse_embed(x):
        if isinstance(x, str):
            # убираем скобки и разбиваем по пробелу
            return np.fromstring(x.strip("[]"), sep=" ", dtype=np.float32)
        elif isinstance(x, (list, np.ndarray)):
            return np.array(x, dtype=np.float32)
        else:
            raise ValueError(f"Неподдерживаемый тип эмбеддинга: {type(x)}")

    embs_list = item_embeddings[emb_col].apply(parse_embed)
    embs = np.stack(embs_list.values)

    embs_tensor = torch.tensor(embs, dtype=torch.float32, device=device)
    embs_tensor = torch.nn.functional.normalize(embs_tensor, dim=1)

    print(f"ANN index built: {len(ids):,} items, embedding dim: {embs_tensor.shape[1]}")

    return ids, embs_tensor


def recommend_topk(model, features, cand_df, K=100):
    """
    model: обученный LightGBM
    features: список фич
    cand_df: dataframe user_id | item_id + фичи
    """
    X = cand_df[features]
    cand_df = cand_df.copy()
    cand_df["score"] = model.predict(
        X, num_iteration=getattr(model, "_best_iteration", None)
    )

    # сортируем внутри каждого пользователя и берём top-K
    topk = (
        cand_df.sort_values(["user_id", "score"], ascending=[True, False])
        .groupby("user_id", group_keys=False)
        .head(K)
        .reset_index(drop=True)
    )
    return topk[["user_id", "item_id", "score"]]


def make_topk(model, features, train_df, K=100, path="submission.csv"):
    """
    Формируем top-K рекомендации и записываем в submission:
    одна строка на пользователя, все K item_id через пробел
    """
    topk_df = recommend_topk(model, features, train_df, K=K)
    topk_df = ensure_k_recs(topk_df, K=K)

    submission_data = [
        {
            "user_id": user_id,
            f"item_id_1 item_id_2 ... item_id_{K}": " ".join(
                map(str, group["item_id"].tolist())
            ),
        }
        for user_id, group in topk_df.groupby("user_id")
    ]

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(path, index=False)
    return submission_df


def ensure_k_recs(df, K=100):
    """
    Гарантируем ровно K рекомендаций на пользователя.
    """
    recs = []
    for user_id, group in df.groupby("user_id"):
        items = group["item_id"].tolist()
        scores = group.get("score", pd.Series([1.0] * len(items))).tolist()
        while len(items) < K:
            items += items
            scores += scores
        recs.append(
            pd.DataFrame(
                {
                    "user_id": [user_id] * K,
                    "item_id": items[:K],
                    "score": scores[:K],
                }
            )
        )
    return pd.concat(recs, ignore_index=True)


if __name__ == "__main__":
    # === 1) Загрузка данных через Dask ===
    print("=== 1) Загрузка данных через Dask ===")
    with ProgressBar():
        orders_ddf, tracker_ddf, items_ddf, category_ddf, test_users_ddf = (
            load_train_data()
        )
    print("Данные загружены.")

    # === 2) Фильтрация товаров по категориям ===
    print("=== 2) Фильтрация товаров по категориям ===")
    items_filtered_ddf = filter_items_by_category(items_ddf, category_ddf)
    items_filtered = items_filtered_ddf.compute()
    print(f"Оставлено товаров: {len(items_filtered):,}")

    # === 3) Взаимодействия пользователей ===
    print("=== 3) Построение взаимодействий пользователей ===")
    interactions_ddf = build_user_interactions(
        orders_ddf, tracker_ddf, items_filtered_ddf
    )
    interactions_df = interactions_ddf.compute().sort_values("timestamp")
    print(f"Взаимодействия построены, всего строк: {len(interactions_df):,}")

    # === 4) Популярные товары (топ 2000) ===
    print("=== 4) Вычисление популярных товаров ===")
    top_pop_df = compute_popularity(interactions_ddf, top_k=2000, window_days=90)
    print("Популярные товары готовы.")

    # === 5) Кандидаты из последних действий пользователя ===
    print("=== 5) Кандидаты из последних действий пользователя ===")
    test_users_list = test_users_ddf["user_id"].compute().tolist()
    recent_cands = user_recent_candidates_dask(
        interactions_ddf,
        test_users_list,
        top_k_per_user=100,
        last_n=30,
        popular_df=top_pop_df,
    )
    print("Кандидаты из последних действий сформированы.")

    # === 6) Co-visitation ===
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

    # === 7) ANN через эмбеддинги (PyTorch) ===
    print("=== 7) ANN через эмбеддинги ===")
    ids, embs_tensor = build_ann_index(
        items_filtered[["item_id", "fclip_embed"]], device="cpu"
    )

    # Формируем user_history словарь
    print("=== Формируем истории пользователей ===")
    user_histories = {
        uid: interactions_df[interactions_df["user_id"] == uid]["item_id"].tolist()
        for uid in tqdm(test_users_list, desc="User histories")
    }

    # ANN кандидаты
    print("=== 7b) Поиск ANN кандидатов ===")
    ann_cands = {}
    batch_size = 500
    top_k_ann = 100
    last_n = 5

    uids = list(user_histories.keys())
    for i in tqdm(range(0, len(uids), batch_size), desc="ANN batches"):
        batch_uids = uids[i : i + batch_size]
        batch_query_embs = []
        batch_map = []

        for uid in batch_uids:
            his = user_histories[uid][-last_n:] if last_n else user_histories[uid]
            for iid in his:
                idx = np.where(ids == iid)[0]
                if idx.size > 0:
                    batch_query_embs.append(embs_tensor[idx[0]])
                    batch_map.append(uid)

        if not batch_query_embs:
            continue

        batch_query_tensor = torch.stack(batch_query_embs)
        batch_query_tensor = torch.nn.functional.normalize(batch_query_tensor, dim=1)
        scores = batch_query_tensor @ embs_tensor.T
        topk_inds = torch.topk(scores, k=top_k_ann, dim=1).indices.cpu().numpy()

        for uid, inds in zip(batch_map, topk_inds):
            if uid not in ann_cands:
                ann_cands[uid] = []
            for ix in inds:
                iid = ids[ix]
                if iid not in user_histories[uid] and iid not in ann_cands[uid]:
                    ann_cands[uid].append(iid)
                if len(ann_cands[uid]) >= top_k_ann:
                    break

    print(f"ANN кандидаты сформированы для {len(ann_cands):,} пользователей")

    # === 8) Объединяем кандидатов ===
    print("=== 8) Объединяем кандидатов ===")
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

    # === 9) Формируем train set для ранжировщика ===
    print("=== 9) Формируем train set ===")
    val_cutoff = interactions_df["timestamp"].quantile(0.9)
    val_data = interactions_df[interactions_df["timestamp"] >= val_cutoff]
    train_df = build_train_set(final_candidates, interactions_df, val_data)
    print(f"Размер train set: {train_df.shape}")

    # === 10) Добавляем фичи ===
    print("=== 10) Добавляем фичи ===")
    orders_df = orders_ddf.compute()
    items_df = items_filtered

    # Популярность категорий
    cat_pop = (
        orders_df.merge(items_df[["item_id", "catalogid"]], on="item_id", how="left")
        .groupby("catalogid")["user_id"]
        .count()
    )
    print(
        f"Популярность категорий рассчитана, уникальных категорий: {cat_pop.shape[0]}"
    )

    # Словарь co-visitation
    covis_dict = {
        (row.item_id_x, row.item_id_y): row.weight
        for row in co_mat_df.itertuples(index=False)
    }
    print(f"Словарь co-visitation содержит {len(covis_dict)} пар товаров")

    train_df = add_features(train_df, orders_df, items_df, cat_pop, covis_dict)
    print("Фичи добавлены.")

    # === 11) Обучаем ранжировщик ===
    print("=== 11) Обучаем модель ранжировщика ===")
    model, features = train_ranker(train_df)
    print("Ранжировщик обучен.")

    # === 12) Top-K рекомендации ===
    print("=== 12) Формируем top-K рекомендации ===")
    topk_df = recommend_topk(model, features, train_df, K=100)
    topk_df = ensure_k_recs(topk_df, K=100)
    print("Top-K рекомендации готовы.")

    # === 13) Submission ===
    print("=== 13) Создаём файл submission ===")
    submission_df = make_topk(model, features, train_df, K=100, path="submission.csv")
    print("Submission готова.")
    print(submission_df.head())

from collections import Counter

import dask.dataframe as dd
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


# -------------------- Загрузка данных --------------------
def load_train_data(max_parts=0, max_rows=0):
    print("=== Загрузка данных через Dask ===")
    paths = {
        "orders": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_orders_data/*/*.parquet",
        "tracker": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_tracker_data/*/*.parquet",
        "items": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_items_data/*.parquet",
        "categories": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train_final_categories_tree/*.parquet",
        "test_users": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_test_for_participants/test_for_participants/*.parquet",
    }

    def read_sample(path):
        ddf = dd.read_parquet(path)
        if max_parts > 0:
            ddf = ddf.partitions[: min(ddf.npartitions, max_parts)]
        if max_rows > 0:
            sample_df = ddf.head(max_rows, compute=True)
            ddf = dd.from_pandas(sample_df, npartitions=1)
        return ddf

    orders_ddf = read_sample(paths["orders"])
    tracker_ddf = read_sample(paths["tracker"])
    items_ddf = read_sample(paths["items"])
    categories_ddf = dd.read_parquet(paths["categories"])
    test_users_ddf = dd.read_parquet(paths["test_users"])

    print(
        f"Orders: {orders_ddf.shape[0].compute():,}\n Tracker: {tracker_ddf.shape[0].compute():,}\n Items: {items_ddf.shape[0].compute():,}\n Categories: {categories_ddf.shape[0].compute():,}\n Test users: {test_users_ddf.shape[0].compute():,}"
    )
    return orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_ddf


# -------------------- Фильтрация данных --------------------
def filter_data(orders_ddf, tracker_ddf, items_ddf):
    orders_ddf = orders_ddf[orders_ddf["last_status"] == "delivered_orders"]
    allowed_actions = ["page_view", "favorite", "to_cart"]
    tracker_ddf = tracker_ddf[tracker_ddf["action_type"].isin(allowed_actions)]
    items_ddf = items_ddf.drop(columns=["variant_id", "model_id"], errors="ignore")
    print("Данные отфильтрованы")
    return orders_ddf, tracker_ddf, items_ddf


# -------------------- Рекомендации --------------------
def recommend_recent_purchases(orders_ddf, test_users_ddf, top_k=100, recent_n=5):
    """
    Генерация рекомендаций:
    - последние покупки пользователя с удвоенным весом для последних N
    - дополнение глобальными популярными товарами
    """
    orders_df = orders_ddf.compute().sort_values(["user_id", "created_timestamp"])
    test_users_df = test_users_ddf.compute()

    # --- Считаем последние покупки пользователей ---
    user_items_weighted = {}
    for user_id, group in tqdm(orders_df.groupby("user_id"), desc="User profiles"):
        items = group["item_id"].tolist()
        weights = [1] * len(items)
        if len(items) > recent_n:
            for i in range(-recent_n, 0):
                weights[i] *= 2
        counter = Counter()
        for item, w in zip(items, weights):
            counter[item] += w
        user_items_weighted[user_id] = counter

    # --- Глобальные популярные товары ---
    popular_items = orders_df["item_id"].value_counts().index.tolist()

    # --- Формируем рекомендации ---
    recommendations = {}
    for user_id in test_users_df["user_id"].unique():
        counter = user_items_weighted.get(user_id, Counter())
        top_items = [item for item, _ in counter.most_common(top_k)]
        # дополняем популярными
        if len(top_items) < top_k:
            for item in popular_items:
                if item not in top_items:
                    top_items.append(item)
                if len(top_items) >= top_k:
                    break
        recommendations[user_id] = top_items

    return recommendations


# -------------------- Сохранение --------------------
def save_submission_csv(recommendations, top_k=100, filename="submission.csv"):
    submission_data = []
    for user_id, items in recommendations.items():
        submission_data.append(
            {
                "user_id": user_id,
                f"item_id_1 item_id_2 ... item_id_{top_k}": " ".join(map(str, items)),
            }
        )
    pd.DataFrame(submission_data).to_csv(filename, index=False)
    print(f"CSV сохранен как {filename}")


# -------------------- Пример использования --------------------
if __name__ == "__main__":
    K = 100
    recent_n = 4

    # 1. Загрузка
    orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_ddf = (
        load_train_data()
    )

    # 2. Фильтрация
    orders_ddf, tracker_ddf, items_ddf = filter_data(orders_ddf, tracker_ddf, items_ddf)

    # 3. Генерация рекомендаций (последние покупки + популярные)
    recommendations = recommend_recent_purchases(
        orders_ddf, test_users_ddf, top_k=K, recent_n=recent_n
    )

    # 4. Сохранение
    save_submission_csv(recommendations, top_k=K)

from collections import Counter

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

import faiss

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

    # Указываем только нужные колонки для каждого датасета
    columns_map = {
        "orders": [
            "item_id",
            "user_id",
            "created_timestamp",
            "last_status",
            "last_status_timestamp",
        ],
        "tracker": ["item_id", "user_id", "timestamp", "action_type", "action_widget"],
        "items": [
            "item_id",
            "itemname",
            "attributes",
            "fclip_embed",
            "catalogid",
            "variant_id",
            "model_id",
        ],
        "categories": ["catalogid", "catalogpath", "ids"],
        "test_users": ["user_id"],
    }

    def read_sample(path, columns=None):
        ddf = dd.read_parquet(path, columns=columns)
        if max_parts > 0:
            ddf = ddf.partitions[: min(ddf.npartitions, max_parts)]
        if max_rows > 0:
            sample_df = ddf.head(max_rows, compute=True)
            ddf = dd.from_pandas(sample_df, npartitions=1)
        return ddf

    orders_ddf = read_sample(paths["orders"], columns=columns_map["orders"])
    tracker_ddf = read_sample(paths["tracker"], columns=columns_map["tracker"])
    items_ddf = read_sample(paths["items"], columns=columns_map["items"])
    categories_ddf = read_sample(paths["categories"], columns=columns_map["categories"])
    test_users_ddf = read_sample(paths["test_users"], columns=columns_map["test_users"])

    print(
        f"Orders: {orders_ddf.shape[0].compute():,}\n"
        f"Tracker: {tracker_ddf.shape[0].compute():,}\n"
        f"Items: {items_ddf.shape[0].compute():,}\n"
        f"Categories: {categories_ddf.shape[0].compute():,}\n"
        f"Test users: {test_users_ddf.shape[0].compute():,}"
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


# -------------------- Считаем взаимодействия пользователя --------------------
def compute_user_interactions(tracker_ddf, action_weights=None):
    """
    Быстрый батчевый подсчет интереса пользователя к товарам на CPU с Dask.
    Возвращает словарь: user_id -> Counter({item_id: суммарный вес})
    """
    if action_weights is None:
        action_weights = {"page_view": 1, "favorite": 2, "to_cart": 3}

    print("-> Присваиваем веса действиям...")
    tracker_ddf = tracker_ddf.assign(
        weight=tracker_ddf["action_type"].map(action_weights).fillna(0)
    )

    print("-> Группируем по user_id и item_id и суммируем веса...")
    with ProgressBar():
        grouped = (
            tracker_ddf.groupby(["user_id", "item_id"])["weight"]
            .sum()
            .compute()  # Dask агрегирует партициями и выводит pandas Series
        )

    # Сохраняем промежуточный результат на диск
    print("-> Сохраняем результат в Parquet...")
    grouped.reset_index().to_parquet(
        "/home/root6/python/e_cup/rec_system/data/processed/user_item_weights.parquet"
    )

    print("-> Преобразуем в словарь user_id -> Counter({item_id: weight})...")
    result = {}
    for (user, item), w in grouped.items():
        if user not in result:
            result[user] = Counter()
        result[user][item] = w

    print("-> Завершено.")
    return result


# -------------------- Генерация рекомендаций --------------------
def recommend_user_based(
    orders_ddf, tracker_ddf, test_users_ddf, top_k=100, recent_n=5
):
    """
    Рекомендации с учетом:
    - последних покупок пользователя
    - частоты взаимодействий с товарами (tracker)
    - дополнение глобальными популярными товарами
    """

    print("=== Начинаем вычисление рекомендаций ===")

    # --- 1. Последние покупки ---
    print("-> Обрабатываем последние покупки пользователей...")
    orders_df = orders_ddf.compute().sort_values(["user_id", "created_timestamp"])
    test_users_df = test_users_ddf.compute()
    user_items_weighted = {}
    for user_id, group in tqdm(orders_df.groupby("user_id"), desc="Последние покупки"):
        items = group["item_id"].tolist()
        weights = [1] * len(items)
        if len(items) > recent_n:
            for i in range(-recent_n, 0):
                weights[i] *= 2
        counter = Counter()
        for item, w in zip(items, weights):
            counter[item] += w
        user_items_weighted[user_id] = counter
    print("-> Последние покупки пользователей обработаны")

    # --- 2. Интерес пользователя через tracker ---
    print("-> Считаем взаимодействия пользователя с товарами (tracker)...")
    user_interactions = compute_user_interactions(
        tracker_ddf
    )  # предполагается, что функция уже есть
    for user_id, counter in tqdm(
        user_interactions.items(), desc="Интерес пользователя"
    ):
        if user_id in user_items_weighted:
            user_items_weighted[user_id].update(counter)
        else:
            user_items_weighted[user_id] = counter
    print("-> Интерес пользователя учтен")

    # --- 3. Глобальные популярные товары ---
    print("-> Формируем список глобально популярных товаров...")
    popular_items = orders_df["item_id"].value_counts().index.tolist()
    print("-> Глобальные популярные товары сформированы")

    # --- 4. Формируем рекомендации ---
    print("-> Генерируем рекомендации для пользователей...")
    recommendations = {}
    for user_id in tqdm(
        test_users_df["user_id"].unique(), desc="Формирование рекомендаций"
    ):
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
    print("-> Рекомендации сформированы")

    print("=== Завершено ===")
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
    recent_n = 20

    # 1. Загрузка
    print("Загрузка данных")
    orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_ddf = (
        load_train_data()
    )

    # 2. Фильтрация
    print("Фильтрация данных")
    orders_ddf, tracker_ddf, items_ddf = filter_data(orders_ddf, tracker_ddf, items_ddf)

    # 3. Рекомендации с учетом последних покупок и взаимодействий
    print("Подбор рекомендаций")
    recommendations = recommend_user_based(
        orders_ddf, tracker_ddf, test_users_ddf, top_k=K, recent_n=recent_n
    )

    # 4. Сохранение
    save_submission_csv(recommendations, top_k=K)

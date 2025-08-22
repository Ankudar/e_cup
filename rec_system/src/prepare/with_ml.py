import os
import pickle
from collections import Counter

import dask.dataframe as dd
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from tqdm.auto import tqdm

tqdm.pandas()


# -------------------- Загрузка данных --------------------
def load_train_data(max_parts=0, max_rows=10_000_000):
    paths = {
        "orders": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_orders_data/*/*.parquet",
        "tracker": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_tracker_data/*/*.parquet",
        "items": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_items_data/*.parquet",
        "test_users": "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_test_for_participants/test_for_participants/*.parquet",
    }

    columns_map = {
        "orders": ["item_id", "user_id", "created_timestamp", "last_status"],
        "tracker": ["item_id", "user_id", "timestamp", "action_type"],
        "items": ["item_id", "itemname", "fclip_embed", "catalogid"],
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

    orders_ddf = read_sample(paths["orders"], columns_map["orders"])
    tracker_ddf = read_sample(paths["tracker"], columns_map["tracker"])
    items_ddf = read_sample(paths["items"], columns_map["items"])
    test_users_ddf = read_sample(paths["test_users"], columns_map["test_users"])

    return orders_ddf, tracker_ddf, items_ddf, test_users_ddf


# -------------------- Фильтрация данных --------------------
def filter_data(orders_ddf, tracker_ddf, items_ddf):
    orders_ddf = orders_ddf[orders_ddf["last_status"] == "delivered_orders"]
    allowed_actions = ["page_view", "favorite", "to_cart"]
    tracker_ddf = tracker_ddf[tracker_ddf["action_type"].isin(allowed_actions)]
    return orders_ddf, tracker_ddf, items_ddf


# -------------------- Подготовка взаимодействий --------------------
def prepare_interactions(orders_ddf, tracker_ddf, action_weights=None):
    if action_weights is None:
        action_weights = {"page_view": 1, "favorite": 2, "to_cart": 3}

    # Последние покупки
    orders_df = orders_ddf.compute()
    orders_df["weight"] = 5  # Покупка сильнее всех действий

    # Действия пользователей
    tracker_df = tracker_ddf.compute()
    tracker_df["weight"] = tracker_df["action_type"].map(action_weights).fillna(0)

    interactions_df = pd.concat(
        [
            orders_df[["user_id", "item_id", "weight"]],
            tracker_df[["user_id", "item_id", "weight"]],
        ],
        ignore_index=True,
    )

    # Суммируем веса
    interactions_df = (
        interactions_df.groupby(["user_id", "item_id"])["weight"].sum().reset_index()
    )
    return interactions_df


# -------------------- Обучение модели --------------------
def train_lightfm(interactions_df):
    dataset = Dataset()
    dataset.fit(
        users=interactions_df["user_id"].unique(),
        items=interactions_df["item_id"].unique(),
    )

    (interactions, _) = dataset.build_interactions(
        [(row.user_id, row.item_id, row.weight) for row in interactions_df.itertuples()]
    )

    model = LightFM(no_components=64, loss="warp")
    model.fit(interactions, epochs=10, num_threads=4)
    return model, dataset


# -------------------- Генерация гибридных рекомендаций --------------------
def generate_recommendations_hybrid(
    model, dataset, interactions_df, items_df, test_users_ddf, top_k=100, recent_n=5
):
    test_users_df = test_users_ddf.compute()
    interactions_df = interactions_df.copy()
    items_df = items_df.set_index("item_id")

    # Последние покупки пользователя
    user_recent = (
        interactions_df.sort_values("weight", ascending=False)
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )

    n_items = dataset.num_items()
    inv_item_map = {v: k for k, v in dataset.mapping()[2].items()}
    recommendations = {}

    for user_id in tqdm(
        test_users_df["user_id"].tolist(), desc="Формируем рекомендации"
    ):
        user_x = dataset.mapping()[0].get(user_id)
        if user_x is None:
            # Cold-start: топ популярных
            top_items = (
                interactions_df.groupby("item_id")["weight"]
                .sum()
                .sort_values(ascending=False)
                .index[:top_k]
                .tolist()
            )
            recommendations[user_id] = top_items
            continue

        # --- 1. Предсказания LightFM ---
        scores = model.predict(user_x, np.arange(n_items))
        item_indices = np.argsort(-scores)
        top_items = [inv_item_map[i] for i in item_indices]

        # --- 2. Усиление последних покупок ---
        recent_items = user_recent.get(user_id, [])[-recent_n:]
        for item in recent_items:
            if item in top_items:
                top_items.remove(item)
            top_items = [item] + top_items  # сдвигаем наверх

        # --- 3. Добавление похожих товаров по категории ---
        augmented = []
        for item in top_items:
            if item not in items_df.index:
                continue
            catalog_id = items_df.loc[item, "catalogid"]
            related = items_df[items_df["catalogid"] == catalog_id].index.tolist()
            related = [i for i in related if i != item]
            augmented.extend(related)

        # Формируем финальный топ_k
        final_top = []
        seen = set()
        for item in top_items + augmented:
            if item not in seen:
                final_top.append(item)
                seen.add(item)
            if len(final_top) >= top_k:
                break

        recommendations[user_id] = final_top[:top_k]

    return recommendations


# -------------------- Сохранение модели и рекомендаций --------------------
def save_model(model, dataset, path="model_lightfm.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "dataset": dataset}, f)
    print(f"Модель сохранена в {path}")


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


# -------------------- Основной запуск --------------------
if __name__ == "__main__":
    K = 100
    RECENT_N = 5

    # 1. Загрузка
    orders_ddf, tracker_ddf, items_ddf, test_users_ddf = load_train_data()

    # 2. Фильтрация
    orders_ddf, tracker_ddf, items_ddf = filter_data(orders_ddf, tracker_ddf, items_ddf)

    # 3. Подготовка interactions
    interactions_df = prepare_interactions(orders_ddf, tracker_ddf)

    # 4. Обучение модели
    model, dataset = train_lightfm(interactions_df)
    save_model(model, dataset, path="model_lightfm.pkl")

    # 5. Генерация гибридных рекомендаций
    items_df_pd = items_ddf.compute()
    recommendations = generate_recommendations_hybrid(
        model,
        dataset,
        interactions_df,
        items_df_pd,
        test_users_ddf,
        top_k=K,
        recent_n=RECENT_N,
    )

    # 6. Сохранение CSV
    save_submission_csv(recommendations, top_k=K, filename="submission.csv")

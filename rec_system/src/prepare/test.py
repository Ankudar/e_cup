import dask
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from dask.diagnostics import ProgressBar  # type: ignore
from tqdm import tqdm
from tqdm.auto import tqdm

tqdm.pandas()

device = "cuda" if torch.cuda.is_available() else "cpu"

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def load_train_data(max_parts=0, max_rows=0):
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
        ddf = dd.read_parquet(path)
        if max_parts > 0:
            n_parts = min(ddf.npartitions, max_parts)
            ddf = ddf.partitions[:n_parts]
        if max_rows > 0:
            sample_df = ddf.head(max_rows, compute=True)
            ddf = dd.from_pandas(sample_df, npartitions=1)
        return ddf

    def read_full(path):
        return dd.read_parquet(path)

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

    categories_ddf = read_full(paths["categories"])
    print(f"Категории: {categories_ddf.shape[0].compute():,} строк (полный прогон)")

    test_users_ddf = read_full(paths["test_users"])
    print(
        f"Тестовые пользователи: {test_users_ddf.shape[0].compute():,} строк (полный прогон)"
    )

    return orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_ddf


# --- Baseline на популярных товарах ---
def compute_popular_items(orders_ddf, top_k=100):
    """Возвращает топ-K популярных товаров"""
    orders_df = orders_ddf.compute()
    item_counts = orders_df.groupby("item_id").size().sort_values(ascending=False)
    top_items = item_counts.head(top_k).index.tolist()
    return top_items


def recommend_popular_for_users(test_users_ddf, popular_items):
    """Для каждого пользователя возвращаем один и тот же топ популярных товаров"""
    test_users_df = test_users_ddf.compute()
    recommendations = {}
    for user_id in test_users_df["user_id"].unique():
        recommendations[user_id] = popular_items
    return recommendations


def save_submission_csv(recommendations, top_k=100, filename="submission.csv"):
    """Сохраняем рекомендации в CSV в нужном формате"""
    submission_data = []
    for user_id, items in recommendations.items():
        submission_data.append(
            {
                "user_id": user_id,
                f"item_id_1 item_id_2 ... item_id_{top_k}": " ".join(map(str, items)),
            }
        )
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(filename, index=False)
    print(f"CSV сохранен как {filename}")


# --- Пример использования ---
if __name__ == "__main__":
    K = 100
    orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_ddf = (
        load_train_data()
    )

    top_items = compute_popular_items(orders_ddf, top_k=K)
    recommendations = recommend_popular_for_users(test_users_ddf, top_items)
    save_submission_csv(recommendations, top_k=K)

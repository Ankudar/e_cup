import ast
import gc
import glob
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

import dask
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def load_train_data():
    print("Загружаем тренировочные данные через Dask...")

    orders_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_orders_data/*/*.parquet"
    tracker_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_tracker_data/*/*.parquet"
    items_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train/final_apparel_items_data/*.parquet"
    categories_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_train_final_categories_tree/*.parquet"
    test_users_path = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_test_for_participants/test_for_participants/*.parquet"

    # --- Загружаем заказы ---
    orders_ddf = dd.read_parquet(orders_path)
    print(f"Найдено файлов заказов: {orders_ddf.npartitions} частей")

    # --- Загружаем взаимодействия ---
    tracker_ddf = dd.read_parquet(tracker_path)
    print(f"Найдено файлов взаимодействий: {tracker_ddf.npartitions} частей")

    # --- Загружаем товары ---
    items_ddf = dd.read_parquet(items_path)
    print(f"Найдено файлов товаров: {items_ddf.npartitions} частей")

    # --- Загружаем категории ---
    categories_ddf = dd.read_parquet(categories_path)
    print(f"Категорий после фильтрации: {categories_ddf.shape[0].compute():,}")

    # --- Загружаем тестовых юзеров ---
    test_users_df = dd.read_parquet(test_users_path)
    print(f"Категорий после фильтрации: {test_users_df.shape[0].compute():,}")

    # --- Возвращаем Dask DataFrame ---
    return orders_ddf, tracker_ddf, items_ddf, categories_ddf, test_users_df


orders_df, tracker_df, items_df, category_df, test_users_df = load_train_data()

# ---------- 1) Фильтрация категорий и подготовка items ----------
TARGET_ROOT_IDS: Set[int] = {7500, 17777, 7697}  # Одежда, Обувь, Аксессуары


def catalogids_for_targets(
    category_df: pd.DataFrame, target_root_ids: Set[int]
) -> Set[int]:
    # category_df['ids'] — массив id по пути категории
    def has_target(ids):
        if isinstance(ids, (list, np.ndarray)):
            s = set(int(x) for x in ids)
        else:
            # на случай строкового представления
            try:
                parsed = ast.literal_eval(str(ids))
                s = set(int(x) for x in parsed)
            except Exception:
                return False
        return len(s & target_root_ids) > 0

    mask = category_df["ids"].apply(has_target)
    return set(category_df.loc[mask, "catalogid"].astype(int).tolist())


def parse_embed(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        # Быстрая парсилка из строки вида "[ 0.1 0.2 ...]"
        arr = np.fromstring(s.strip("[]").replace("\n", " "), sep=" ")
        return arr.astype(np.float32)
    # fallback
    try:
        parsed = ast.literal_eval(s)
        return np.asarray(parsed, dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


def prepare_items(items_df: pd.DataFrame, category_df: pd.DataFrame) -> pd.DataFrame:
    allowed_catalogids = catalogids_for_targets(category_df, TARGET_ROOT_IDS)
    items = items_df.copy()
    items["catalogid"] = items["catalogid"].astype(int)
    items = items[items["catalogid"].isin(allowed_catalogids)]
    items["fclip_embed"] = items["fclip_embed"].apply(parse_embed)
    items = items[items["fclip_embed"].apply(lambda v: v.size > 0)]
    items = items.drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    return items[["item_id", "catalogid", "fclip_embed"]]


# ---------- 2) Интеракции: только покупки (позитивы) ----------
POSITIVE_STATUSES = {"delivered_orders"}  # можно расширить при необходимости


def prepare_purchases(orders_df: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    orders = orders_df.copy()
    orders["created_timestamp"] = pd.to_datetime(orders["created_timestamp"])
    if "last_status" in orders.columns:
        orders = orders[orders["last_status"].isin(POSITIVE_STATUSES)]
    # Оставляем только товары из целевых категорий
    orders = orders.merge(items[["item_id"]], on="item_id", how="inner")
    interactions = orders[["user_id", "item_id", "created_timestamp"]].rename(
        columns={"created_timestamp": "ts"}
    )
    interactions["weight"] = 1.0  # покупка как сильный сигнал
    interactions = (
        interactions.drop_duplicates(["user_id", "item_id", "ts"])
        .sort_values("ts")
        .reset_index(drop=True)
    )
    return interactions


# ---------- 3) Leave-one-out сплит по времени ----------
def leave_one_out_split(
    interactions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Для каждого пользователя: последние 2 покупки -> val и test, остальное -> train
    interactions = interactions.sort_values(["user_id", "ts"])

    def mark_phase(dfu):
        n = len(dfu)
        if n == 1:
            dfu["phase"] = "train"
        elif n == 2:
            dfu.iloc[:-1, dfu.columns.get_loc("phase")] = "val"
            dfu.iloc[-1, dfu.columns.get_loc("phase")] = "test"
        else:
            dfu.iloc[:-2, dfu.columns.get_loc("phase")] = "train"
            dfu.iloc[-2, dfu.columns.get_loc("phase")] = "val"
            dfu.iloc[-1, dfu.columns.get_loc("phase")] = "test"
        return dfu

    interactions = interactions.copy()
    interactions["phase"] = ""
    interactions = interactions.groupby("user_id", group_keys=False).apply(mark_phase)

    train = interactions[interactions["phase"] == "train"].drop(columns="phase")
    val = interactions[interactions["phase"] == "val"].drop(columns="phase")
    test = interactions[interactions["phase"] == "test"].drop(columns="phase")
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ---------- 4) Метрики для offline-оценки ----------
def recall_at_k(
    y_true: Dict[int, Set[int]], y_pred: Dict[int, list], k: int = 100
) -> float:
    num, den = 0, 0
    for u, true_items in y_true.items():
        den += len(true_items)
        recs = y_pred.get(u, [])[:k]
        num += len(true_items & set(recs))
    return num / den if den > 0 else 0.0


def dcg_at_k(rels: list, k: int) -> float:
    rels = rels[:k]
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(
    y_true: Dict[int, Set[int]], y_pred: Dict[int, list], k: int = 100
) -> float:
    scores = []
    for u, true_items in y_true.items():
        pred = y_pred.get(u, [])[:k]
        rels = [1 if it in true_items else 0 for it in pred]
        dcg = dcg_at_k(rels, k)
        idcg = dcg_at_k(sorted(rels, reverse=True), k) if true_items else 1.0
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# ---------- 5) Пример пайплайна этапа 1 ----------
def stage1_build_splits(
    orders_df: pd.DataFrame, items_df: pd.DataFrame, category_df: pd.DataFrame
):
    items = prepare_items(items_df, category_df)
    interactions = prepare_purchases(orders_df, items)
    train, val, test = leave_one_out_split(interactions)
    # Готовим таргеты для валидации/теста (ground truth множества)
    y_val = (val.groupby("user_id")["item_id"].apply(set)).to_dict()
    y_test = (test.groupby("user_id")["item_id"].apply(set)).to_dict()
    meta = {
        "n_users": interactions["user_id"].nunique(),
        "n_items": items["item_id"].nunique(),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }
    return items, train, val, test, y_val, y_test, meta


# ---------- 6) Отбор тест-юзеров под инференс ----------
TEST_USERS = test_users_df


def users_for_inference(train: pd.DataFrame, test_users: list) -> pd.DataFrame:
    # История для кандидато-генерации по этим пользователям
    return train[train["user_id"].isin([int(u) for u in test_users])].sort_values("ts")

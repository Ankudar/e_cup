# predict_xgboost.py
import gc
import glob
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

# ===== ПУТИ / КОНСТАНТЫ =====
MODEL_PATH = "/home/root6/python/e_cup/rec_system/src/models/model.json"
ARTIFACTS_PATH = "/home/root6/python/e_cup/rec_system/src/models/model.pkl"
TEST_USERS_PATH = "/home/root6/python/e_cup/rec_system/data/raw/test_users/*.parquet"
OUTPUT_PATH = "/home/root6/python/e_cup/rec_system/result/submission.csv"
TOP_K = 100
BATCH_USERS = 1000

# Веса действий
ACTION_WEIGHTS = {"page_view": 2, "favorite": 5, "to_cart": 10}

# ===== ЛОГИ =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("predict_xgboost")


def log_message(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full = f"[{ts}] {msg}"
    print(full)
    sys.stdout.flush()


# ===== УТИЛИТЫ =====
def ensure_dir(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_recommendations_to_csv(
    recs: dict[int, list[int]], output_path: str, header: bool
):
    ensure_dir(output_path)
    mode = "w" if header else "a"
    with open(output_path, mode, encoding="utf-8", buffering=1 << 20) as f:
        if header:
            f.write("user_id,item_id_1 item_id_2 ... item_id_100\n")
        for uid, items in recs.items():
            f.write(f"{int(uid)},{' '.join(map(str, map(int, items)))}\n")


# ===== ОПТИМИЗИРОВАННАЯ ПОДГОТОВКА ФИЧ =====
def make_candidate_frame_batch(
    user_ids: list[int],
    all_candidates: list[list[int]],
    user_features_dict: dict,
    item_features_dict: dict,
    feature_columns: list[str],
    cat_features: list[str],
) -> pd.DataFrame:
    """Обрабатываем несколько пользователей за раз"""
    all_rows = []

    item_names = [c for c in feature_columns if c.startswith("item_")]
    user_names = [c for c in feature_columns if c.startswith("user_")]

    for user_id, candidates in zip(user_ids, all_candidates):
        if not candidates:
            continue

        # Фичи пользователя
        u = user_features_dict.get(user_id, {})
        if isinstance(u, np.ndarray):
            u = {
                name: float(u[idx]) if idx < len(u) else 0.0
                for idx, name in enumerate(user_names)
            }
        elif not isinstance(u, dict):
            u = {}

        user_part = [float(u.get(name, 0.0)) for name in user_names]

        # Для всех кандидатов пользователя
        for iid in candidates:
            it = item_features_dict.get(iid, {})
            if isinstance(it, np.ndarray):
                it = {
                    name: float(it[idx]) if idx < len(it) else 0.0
                    for idx, name in enumerate(item_names)
                }
            elif not isinstance(it, dict):
                it = {}

            item_part = [float(it.get(name, 0.0)) for name in item_names]
            all_rows.append(user_part + item_part + [user_id, iid])

    if not all_rows:
        return pd.DataFrame(columns=feature_columns + ["user_id", "item_id"])

    df = pd.DataFrame(
        all_rows, columns=user_names + item_names + ["user_id", "item_id"]
    )
    df = df.reindex(columns=feature_columns + ["user_id", "item_id"], fill_value=0.0)

    # Обработка категориальных признаков
    for col in cat_features:
        if col in df.columns:
            df[col] = pd.factorize(df[col].fillna("nan"))[0].astype(np.int32)

    for col in df.columns:
        if col not in cat_features and col not in ["user_id", "item_id"]:
            df[col] = df[col].astype("float32")

    return df


# ===== ОПТИМИЗИРОВАННАЯ ГЕНЕРАЦИЯ КАНДИДАТОВ =====
def build_candidates_for_user_fast(
    user_id: int,
    recent_items_get,
    popular_items_array: np.ndarray,
    copurchase_map: dict[int, list[int]],
    item_to_cat: dict[int, int],
    cat_to_items: dict[int, list[int]],
    item_map: set,
    max_candidates: int = 200,
) -> list[int]:
    user_history = recent_items_get(user_id, [])

    if not user_history:
        return popular_items_array[:max_candidates].tolist()

    excluded_items = set()
    item_weights = {}

    for item_action in user_history[:100]:
        if isinstance(item_action, tuple) and len(item_action) >= 2:
            item_id, action_type = item_action[0], item_action[1]
            excluded_items.add(item_id)
            item_weights[item_id] = item_weights.get(item_id, 0) + ACTION_WEIGHTS.get(
                action_type, 1
            )
        else:
            excluded_items.add(item_action)
            item_weights[item_action] = item_weights.get(item_action, 0) + 1

    cands = set()
    recent_items = list(excluded_items)[:100]

    # 1. Сопутствующие товары
    for item_id in recent_items:
        if item_id in copurchase_map:
            weight = item_weights.get(item_id, 1)
            n_to_take = min(15, int(10 * (weight / 5)))

            for candidate in copurchase_map[item_id][:n_to_take]:
                if candidate not in excluded_items and candidate in item_map:
                    cands.add(candidate)
                    if len(cands) >= max_candidates:
                        return list(cands)

    # 2. Товары из категорий
    for item_id in recent_items[:100]:
        cat = item_to_cat.get(item_id)
        if cat and cat in cat_to_items:
            weight = item_weights.get(item_id, 1)
            n_to_take = min(12, int(8 * (weight / 5)))

            for candidate in cat_to_items[cat][:n_to_take]:
                if candidate not in excluded_items and candidate in item_map:
                    cands.add(candidate)
                    if len(cands) >= max_candidates:
                        return list(cands)

    # 3. Популярные товары
    if len(cands) < max_candidates:
        for candidate in popular_items_array:
            if candidate not in excluded_items and candidate in item_map:
                cands.add(candidate)
                if len(cands) >= max_candidates:
                    break

    return list(cands)[:max_candidates]


# ===== ОСНОВНОЙ ИНФЕРЕНС =====
def main():
    start_time = time.time()
    log_message("=== Старт инференса XGBoost ===")

    # --- Загрузка пользователей
    log_message("Загрузка пользователей...")

    # Используем glob для получения списка файлов
    parquet_files = glob.glob(TEST_USERS_PATH)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found matching: {TEST_USERS_PATH}")

    # Читаем все файлы последовательно
    test_dfs = []
    for file_path in parquet_files:
        test_dfs.append(pd.read_parquet(file_path))

    test_df = pd.concat(test_dfs, ignore_index=True)
    test_users = test_df["user_id"].astype(np.int64).unique().tolist()
    log_message(
        f"Загружено {len(test_users)} пользователей из {len(parquet_files)} файлов"
    )

    # --- Загрузка модели и артефактов
    log_message("Загрузка модели...")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)

    log_message("Загрузка артефактов...")
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)

    feature_columns = artifacts.get("feature_columns", [])
    cat_features = artifacts.get("cat_features", [])

    item_map_set = set(artifacts["item_map"].keys())
    popular_items_array = np.array(artifacts["popular_items"], dtype=np.int64)

    log_message(f"Загружено {len(feature_columns)} признаков")

    # --- Генерация рекомендаций батчами
    recommendations = {}
    header_written = False
    processed = 0

    log_message("Начинаем генерацию рекомендаций...")

    with tqdm(total=len(test_users), desc="Ранжирование") as pbar:
        for i in range(0, len(test_users), BATCH_USERS):
            batch_users = test_users[i : i + BATCH_USERS]
            batch_candidates = []
            batch_user_ids = []

            # Генерируем кандидатов для батча
            for uid in batch_users:
                try:
                    cands = build_candidates_for_user_fast(
                        user_id=uid,
                        recent_items_get=artifacts["recent_items_map"].get,
                        popular_items_array=popular_items_array,
                        copurchase_map=artifacts["copurchase_map"],
                        item_to_cat=artifacts["item_to_cat"],
                        cat_to_items=artifacts.get("cat_to_items", {}),
                        item_map=item_map_set,
                        max_candidates=200,
                    )
                    batch_candidates.append(cands)
                    batch_user_ids.append(uid)
                except Exception as e:
                    batch_candidates.append(popular_items_array[:TOP_K].tolist())
                    batch_user_ids.append(uid)

            # Обрабатываем батч
            if batch_user_ids:
                try:
                    X_batch = make_candidate_frame_batch(
                        user_ids=batch_user_ids,
                        all_candidates=batch_candidates,
                        user_features_dict=artifacts["user_features_dict"],
                        item_features_dict=artifacts["item_features_dict"],
                        feature_columns=feature_columns,
                        cat_features=cat_features,
                    )

                    if not X_batch.empty:
                        X_pred = X_batch[feature_columns]
                        dmatrix = xgb.DMatrix(X_pred)
                        probs = model.predict(dmatrix)

                        # ОПТИМИЗИРОВАННАЯ ЧАСТЬ - без фрагментации DataFrame
                        result_df = pd.DataFrame(
                            {
                                "user_id": X_batch["user_id"].values,
                                "item_id": X_batch["item_id"].values,
                                "score": probs,
                            }
                        )

                        top_items_per_user = {}
                        for user_id, group in result_df.groupby("user_id"):
                            # Берем TOP_K items с наибольшими score
                            top_items = group.nlargest(TOP_K, "score")[
                                "item_id"
                            ].tolist()

                            # Добираем популярными если нужно
                            if len(top_items) < TOP_K:
                                user_history = artifacts["recent_items_map"].get(
                                    user_id, []
                                )
                                excluded = set()
                                for item_action in user_history:
                                    if isinstance(item_action, tuple):
                                        excluded.add(item_action[0])
                                    else:
                                        excluded.add(item_action)

                                for popular_item in popular_items_array:
                                    if (
                                        popular_item not in top_items
                                        and popular_item not in excluded
                                    ):
                                        top_items.append(popular_item)
                                    if len(top_items) >= TOP_K:
                                        break

                            top_items_per_user[user_id] = top_items[:TOP_K]

                        # Добавляем в recommendations
                        for user_id, top_items in top_items_per_user.items():
                            recommendations[user_id] = top_items

                except Exception as e:
                    for uid in batch_user_ids:
                        recommendations[uid] = popular_items_array[:TOP_K].tolist()
                    log_message(f"Ошибка в батче {i}: {e}")

            processed += len(batch_users)
            pbar.update(len(batch_users))

            if processed % 10000 == 0:
                save_recommendations_to_csv(
                    recommendations, OUTPUT_PATH, header=not header_written
                )
                header_written = True
                recommendations.clear()
                gc.collect()
                log_message(f"Обработано {processed} пользователей")

    if recommendations:
        save_recommendations_to_csv(
            recommendations, OUTPUT_PATH, header=not header_written
        )

    dt = time.time() - start_time
    log_message(f"Готово! Время: {timedelta(seconds=int(dt))}")
    log_message(f"Скорость: {len(test_users)/dt:.1f} users/sec")


if __name__ == "__main__":
    main()

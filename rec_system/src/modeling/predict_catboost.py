# predict_catboost.py
import gc
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from tqdm import tqdm

# ===== ПУТИ / КОНСТАНТЫ =====
MODEL_PATH = "/home/root6/python/e_cup/rec_system/src/models/catboost_model.cbm"
ARTIFACTS_PATH = "/home/root6/python/e_cup/rec_system/src/models/model.pkl"
TEST_USERS_PATH = "/home/root6/python/e_cup/rec_system/data/raw/test_users/*.parquet"
OUTPUT_PATH = "/home/root6/python/e_cup/rec_system/result/submission.csv"
TOP_K = 100
BATCH_USERS = 100

# ===== ЛОГИ =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("predict_catboost")


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
    with open(output_path, mode, encoding="utf-8", buffering=1 << 14) as f:
        if header:
            f.write("user_id,item_id_1 item_id_2 ... item_id_100\n")
        for uid, items in recs.items():
            f.write(f"{int(uid)},{' '.join(map(str, map(int, items)))}\n")


# ===== ПОДГОТОВКА ФИЧ ДЛЯ КАНДИДАТОВ =====
def make_candidate_frame_for_user(
    user_id: int,
    candidates: list[int],
    user_features_dict: dict,
    item_features_dict: dict,
    feature_columns: list[str],
    cat_features: list[str],
) -> pd.DataFrame:
    """
    Собираем DataFrame со строками (user_id, item_id) и колонками из feature_columns.
    Предполагается, что feature_columns состоят из имен user_* и item_* признаков
    (например: user_count, ..., item_count, ...).
    """
    if not candidates:
        return pd.DataFrame(columns=feature_columns)

    # Берём фичи пользователя / товара как dict[str -> float]
    u = user_features_dict.get(user_id, {})
    # Быстрая ветка: если хранится np.array — переведём в dict по именам
    if isinstance(u, np.ndarray):
        # На этапе обучения user_* шли первыми в feature_columns
        u_names = [c for c in feature_columns if c.startswith("user_")]
        u = {
            name: float(u[idx]) if idx < len(u) else 0.0
            for idx, name in enumerate(u_names)
        }
    elif not isinstance(u, dict):
        u = {}

    rows = []
    item_names = [c for c in feature_columns if c.startswith("item_")]
    user_names = [c for c in feature_columns if c.startswith("user_")]

    # Значения user_* одинаковые для всех кандидатов, сформируем раз
    user_part = [float(u.get(name, 0.0)) for name in user_names]

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
        # Собираем полный вектор в порядке feature_columns
        # Порядок: сначала все user_*, потом item_* (как в ваших логах)
        vec = user_part + item_part
        rows.append(vec)

    df = pd.DataFrame(rows, columns=user_names + item_names)
    # На случай если в feature_columns есть что-то ещё — приведём к нужному порядку/составу
    df = df.reindex(columns=feature_columns, fill_value=0.0)

    # приведение типов
    for col in df.columns:
        if col in cat_features:
            df[col] = df[col].fillna("nan").astype(str)
        else:
            df[col] = df[col].astype("float32")
    return df


# ===== КАНДИДАТЫ =====
def build_candidates_for_user(
    user_id: int,
    recent_items_get,
    popular_items_array: np.ndarray,
    copurchase_map: dict[int, list[int]],
    item_to_cat: dict[int, int],
    cat_to_items: dict[int, list[int]],
    item_map: dict[int, int],
) -> list[int]:
    recent = recent_items_get(user_id, [])

    N_RECENT = 30
    N_COPURCHASE = 15
    N_CATEGORY = 15
    N_POPULAR = 5

    cands = set()
    cands.update(recent[:N_RECENT])

    for it in recent[:10]:
        cands.update(copurchase_map.get(it, [])[:N_COPURCHASE])

    for it in recent[:5]:
        cat = item_to_cat.get(it)
        if cat is not None and cat in cat_to_items:
            cands.update(cat_to_items[cat][:N_CATEGORY])

    cands.update(popular_items_array[:N_POPULAR])

    # фильтруем по известным item_id
    cands = [c for c in cands if c in item_map]

    # ограничим размер списка (чтобы не взрывать инференс)
    if len(recent) < 3:
        max_cands = 500
    else:
        max_cands = 300

    if len(cands) > max_cands:
        pop_rank = {item: idx for idx, item in enumerate(popular_items_array)}
        cands = sorted(cands, key=lambda x: pop_rank.get(x, 10**9))[:max_cands]

    return cands


# ===== ОСНОВНОЙ ИНФЕРЕНС =====
def main():
    start_time = time.time()
    log_message("=== Старт инференса CatBoost ===")

    # --- Загрузка пользователей
    test_df = dd.read_parquet(TEST_USERS_PATH).compute()
    test_users = test_df["user_id"].astype(np.int64).unique().tolist()
    log_message(f"Загружено {len(test_users)} пользователей")

    # --- Загрузка модели и артефактов
    log_message(f"Загружаем модель из {MODEL_PATH}")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    # вытаскиваем список признаков и категориальных из модели
    feature_columns = model.feature_names_
    cat_feature_indices = model.get_cat_feature_indices()
    cat_features = [feature_columns[i] for i in cat_feature_indices]

    log_message(
        f"Загружено {len(feature_columns)} признаков, "
        f"{len(cat_features)} категориальных"
    )

    # --- артефакты (только мапы и словари, без feature_columns/cat_features)
    log_message(f"Загружаем артефакты из {ARTIFACTS_PATH}")
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)

    recent_items_map = artifacts["recent_items_map"]
    copurchase_map = artifacts["copurchase_map"]
    item_to_cat = artifacts["item_to_cat"]
    user_features_dict = artifacts["user_features_dict"]
    item_features_dict = artifacts["item_features_dict"]
    item_map = artifacts["item_map"]
    popular_items = artifacts["popular_items"]

    # строим cat_to_items
    cat_to_items = {}
    for item_id, cat_id in item_to_cat.items():
        cat_to_items.setdefault(cat_id, []).append(item_id)

    popular_items_array = np.array(popular_items, dtype=np.int64)
    recent_items_get = recent_items_map.get

    # --- Генерация рекомендаций
    recommendations = {}
    header_written = False
    processed = 0

    with tqdm(total=len(test_users), desc="Ранжирование") as pbar:
        for i in range(0, len(test_users), BATCH_USERS):
            batch_users = test_users[i : i + BATCH_USERS]
            for uid in batch_users:
                try:
                    cands = build_candidates_for_user(
                        user_id=uid,
                        recent_items_get=recent_items_get,
                        popular_items_array=popular_items_array,
                        copurchase_map=copurchase_map,
                        item_to_cat=item_to_cat,
                        cat_to_items=cat_to_items,
                        item_map=item_map,
                    )

                    if not cands:
                        recommendations[uid] = popular_items_array[:TOP_K].tolist()
                    else:
                        X = make_candidate_frame_for_user(
                            user_id=uid,
                            candidates=cands,
                            user_features_dict=user_features_dict,
                            item_features_dict=item_features_dict,
                            feature_columns=feature_columns,
                            cat_features=cat_features,
                        )

                        # приведение категориальных к строке
                        for col in cat_features:
                            if col in X.columns:
                                X[col] = X[col].fillna("nan").astype(str)

                        # предикт
                        pool = Pool(
                            X,
                            cat_features=[c for c in cat_features if c in X.columns],
                        )
                        probs = model.predict_proba(pool)[:, 1]

                        order = np.argsort(probs)[::-1]
                        top = [int(cands[j]) for j in order[:TOP_K]]

                        # доклеиваем популярные при нехватке
                        if len(top) < TOP_K:
                            for it in popular_items_array:
                                if it not in top:
                                    top.append(int(it))
                                if len(top) >= TOP_K:
                                    break
                        recommendations[uid] = top

                except Exception as e:
                    # на фатальной ошибке — популярные
                    recommendations[uid] = popular_items_array[:TOP_K].tolist()
                    log_message(f"Ошибка для пользователя {uid}: {e}")

                processed += 1
                pbar.update(1)

                # периодически сбрасываем в файл и чистим память
                if processed % 10000 == 0:
                    save_recommendations_to_csv(
                        recommendations, OUTPUT_PATH, header=not header_written
                    )
                    header_written = True
                    recommendations.clear()
                    gc.collect()

        # финальный сброс
        if recommendations:
            save_recommendations_to_csv(
                recommendations, OUTPUT_PATH, header=not header_written
            )

    dt = time.time() - start_time
    log_message(f"Готово. Итоговый файл: {OUTPUT_PATH}")
    log_message(f"Время: {timedelta(seconds=int(dt))}")


if __name__ == "__main__":
    main()

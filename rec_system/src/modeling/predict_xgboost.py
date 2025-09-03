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
BATCH_USERS = 2000

# Веса действий
ACTION_WEIGHTS = {"page_view": 1, "favorite": 5, "to_cart": 10}

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


def calculate_user_category_preferences(user_history, item_to_cat):
    """Рассчитывает предпочтения пользователя по категориям"""
    cat_weights = defaultdict(float)
    for item_action in user_history:
        if isinstance(item_action, tuple) and len(item_action) >= 2:
            item_id, action_type = item_action[0], item_action[1]
            weight = ACTION_WEIGHTS.get(action_type, 1)
        else:
            item_id = item_action if isinstance(item_action, int) else item_action[0]
            weight = 1

        category = item_to_cat.get(item_id)
        if category:
            cat_weights[category] += weight

    return dict(cat_weights)


def diversify_recommendations(
    top_items,
    scores,
    user_history,
    item_to_cat,
    max_similar_per_category=2,  # Только 2 товара из одной категории
    top_k=100,
):
    """Обеспечивает разнообразие рекомендаций по категориям"""
    if not top_items or len(top_items) <= top_k:
        return top_items[:top_k]

    # Более агрессивная диверсификация
    final_recommendations = []
    used_categories = defaultdict(int)
    used_items = set()

    # Сортируем по score с учетом diversity penalty
    diversified_items = []
    for i, (item_id, score) in enumerate(zip(top_items, scores)):
        category = item_to_cat.get(item_id, -1)
        category_count = used_categories[category]

        # Штрафуем товары из перепредставленных категорий
        diversity_penalty = category_count * 0.3  # Увеличили штраф
        adjusted_score = score * (1 - diversity_penalty)

        diversified_items.append((item_id, adjusted_score, category))

    # Сортируем по скорректированному score
    diversified_items.sort(key=lambda x: x[1], reverse=True)

    # Отбираем с контролем категорий
    for item_id, adjusted_score, category in diversified_items:
        if len(final_recommendations) >= top_k:
            break

        if (
            used_categories[category] < max_similar_per_category
            and item_id not in used_items
        ):
            final_recommendations.append(item_id)
            used_items.add(item_id)
            used_categories[category] += 1

    # Если не набрали достаточно, добавляем лучшие без учета категорий
    if len(final_recommendations) < top_k:
        for item_id, score, category in diversified_items:
            if len(final_recommendations) >= top_k:
                break
            if item_id not in used_items:
                final_recommendations.append(item_id)
                used_items.add(item_id)

    return final_recommendations[:top_k]


def get_exploration_candidates(user_id, artifacts, num_candidates=100):
    """Возвращает редко рекомендуемые товары для исследования"""
    # 1. Товары из длинного хвоста (не популярные)
    all_items = list(artifacts["item_map"].keys())
    popular_set = set(artifacts["popular_items"][:500])  # Топ-500 популярных

    # Исключаем популярные
    long_tail_items = [item for item in all_items if item not in popular_set]

    # 2. Случайная выборка с учетом user_id для воспроизводимости
    random_seed = user_id % 1000
    np.random.seed(random_seed)

    if len(long_tail_items) > num_candidates:
        exploration_items = np.random.choice(
            long_tail_items, num_candidates, replace=False
        )
    else:
        exploration_items = long_tail_items

    # 3. Добавляем немного новых товаров (если есть информация)
    new_items = artifacts.get("new_items", [])
    if new_items:
        new_to_add = min(10, len(new_items))
        exploration_items = (
            list(exploration_items)[: num_candidates - new_to_add]
            + new_items[:new_to_add]
        )

    return exploration_items


def get_user_segment_popular_items(
    user_id, popular_items_array, segments_dict, segment_popular_dict, default_popular
):
    """Возвращает популярные товары для сегмента пользователя"""
    segment = segments_dict.get(user_id, 0)
    segment_popular = segment_popular_dict.get(segment, default_popular)

    # Добавляем немного случайности для разнообразия
    start_idx = (user_id % 20) * (len(segment_popular) // 20)
    return segment_popular[start_idx : start_idx + 100]


# ===== ОПТИМИЗИРОВАННАЯ ПОДГОТОВКА ФИЧ =====
def make_candidate_frame_batch(
    user_ids: list[int],
    all_candidates: list[list[int]],
    user_features_dict: dict,
    item_features_dict: dict,
    feature_columns: list[str],
    cat_features: list[str],
    # Новые параметры
    item_to_cat: dict = None,
    recent_items_map: dict = None,
) -> pd.DataFrame:
    """Обрабатываем несколько пользователей за раз с улучшенными фичами"""
    all_rows = []

    item_names = [c for c in feature_columns if c.startswith("item_")]
    user_names = [c for c in feature_columns if c.startswith("user_")]

    for user_id, candidates in zip(user_ids, all_candidates):
        if not candidates:
            continue

        # Фичи пользователя
        u = user_features_dict.get(user_id, {})
        if u is None or (isinstance(u, dict) and len(u) == 0):
            u = {name: 0.0 for name in user_names}
        elif isinstance(u, np.ndarray):
            u = {
                name: float(u[idx]) if idx < len(u) else 0.0
                for idx, name in enumerate(user_names)
            }
        elif not isinstance(u, dict):
            u = {}

        user_part = [float(u.get(name, 0.0)) for name in user_names]

        # Предпочтения пользователя по категориям (для новых фич)
        user_cat_preferences = {}
        if item_to_cat is not None and recent_items_map is not None:
            user_history = recent_items_map.get(user_id, [])
            user_cat_preferences = calculate_user_category_preferences(
                user_history, item_to_cat
            )

        # Для всех кандидатов пользователя
        for iid in candidates:
            it = item_features_dict.get(iid, {})
            if it is None or (isinstance(it, dict) and len(it) == 0):
                continue
            elif isinstance(it, np.ndarray):
                it = {
                    name: float(it[idx]) if idx < len(it) else 0.0
                    for idx, name in enumerate(item_names)
                }
            elif not isinstance(it, dict):
                it = {}

            item_part = [float(it.get(name, 0.0)) for name in item_names]

            # Добавляем новые фичи для улучшения персонализации
            row = user_part + item_part + [user_id, iid]

            # Добавляем фичу совпадения категории с предпочтениями пользователя
            if item_to_cat is not None:
                item_category = item_to_cat.get(iid, -1)
                cat_match_score = user_cat_preferences.get(item_category, 0.0)
                row.append(cat_match_score)

            all_rows.append(row)

    if not all_rows:
        base_columns = feature_columns + ["user_id", "item_id"]
        if item_to_cat is not None:
            base_columns += ["cat_match_score"]
        return pd.DataFrame(columns=base_columns)

    # Определяем колонки для DataFrame
    base_columns = user_names + item_names + ["user_id", "item_id"]
    if item_to_cat is not None:
        base_columns += ["cat_match_score"]

    df = pd.DataFrame(all_rows, columns=base_columns)

    # Обеспечиваем правильный порядок колонок
    expected_columns = feature_columns + ["user_id", "item_id"]
    if item_to_cat is not None and "cat_match_score" not in feature_columns:
        expected_columns += ["cat_match_score"]

    df = df.reindex(columns=expected_columns, fill_value=0.0)

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
    max_candidates: int = 300,  # УВЕЛИЧИЛИ чтобы гарантировать 100 после фильтрации
    # Настройки
    popular_max_percentage: float = 0.2,
    cold_start_popular_percentage: float = 0.8,
    artifacts: dict = None,
) -> list[int]:

    user_history = recent_items_get(user_id, [])
    is_cold_start = len(user_history) < 3

    # Для cold start - гарантируем достаточно кандидатов
    if is_cold_start:
        max_popular = int(max_candidates * cold_start_popular_percentage)
        start_idx = (user_id % 50) * (max_popular // 2)
        end_idx = start_idx + max_popular
        if end_idx > len(popular_items_array):
            # Берем с запасом из разных частей
            part1 = popular_items_array[start_idx:]
            part2 = popular_items_array[: max_popular - len(part1)]
            part3 = popular_items_array[
                500 : 500 + max_popular - len(part1) - len(part2)
            ]
            candidates = list(part1) + list(part2) + list(part3)
            return candidates[:max_candidates]
        return popular_items_array[start_idx:end_idx].tolist()

    cands = []
    excluded = set()

    # 1. Товары из истории (гарантируем минимум)
    recent_items = []
    for item_action in user_history[:30]:  # До 30 из истории
        if isinstance(item_action, tuple) and len(item_action) >= 2:
            item_id = item_action[0]
        else:
            item_id = item_action if isinstance(item_action, int) else item_action[0]

        if item_id in item_map and item_id not in excluded:
            recent_items.append(item_id)
            excluded.add(item_id)
            if len(recent_items) >= 50:  # Максимум 50 из истории
                break

    cands.extend(recent_items)

    # 2. Сопутствующие товары (гарантируем достаточное количество)
    copurchase_added = 0
    for item_id in recent_items[:15]:
        if item_id in copurchase_map and len(cands) < max_candidates:
            for candidate in copurchase_map[item_id][:8]:
                if candidate not in excluded and candidate in item_map:
                    cands.append(candidate)
                    excluded.add(candidate)
                    copurchase_added += 1
                    if len(cands) >= max_candidates or copurchase_added >= 60:
                        break
        if len(cands) >= max_candidates:
            break

    # 3. Товары из категорий (гарантируем разнообразие)
    category_added = 0
    cat_counts = defaultdict(int)
    for item_id in recent_items[:20]:
        category = item_to_cat.get(item_id)
        if category and category in cat_to_items and cat_counts[category] < 5:
            for candidate in cat_to_items[category][:10]:
                if candidate not in excluded and candidate in item_map:
                    cands.append(candidate)
                    excluded.add(candidate)
                    category_added += 1
                    cat_counts[category] += 1
                    if len(cands) >= max_candidates or category_added >= 80:
                        break
        if len(cands) >= max_candidates:
            break

    # 4. ГАРАНТИРУЕМ что набрали достаточно кандидатов
    if len(cands) < max_candidates:
        needed = max_candidates - len(cands)

        # Добавляем популярные из разных частей списка
        popular_to_add = []

        # Первая часть популярных
        start_idx1 = 50 + (user_id % 100)
        for i in range(
            start_idx1, min(start_idx1 + needed // 2, len(popular_items_array))
        ):
            candidate = popular_items_array[i]
            if candidate not in excluded and candidate in item_map:
                popular_to_add.append(candidate)
                if len(popular_to_add) >= needed:
                    break

        # Вторая часть популярных (если нужно)
        if len(popular_to_add) < needed:
            start_idx2 = 200 + (user_id % 150)
            for i in range(
                start_idx2, min(start_idx2 + needed, len(popular_items_array))
            ):
                candidate = popular_items_array[i]
                if (
                    candidate not in excluded
                    and candidate in item_map
                    and candidate not in popular_to_add
                ):
                    popular_to_add.append(candidate)
                    if len(popular_to_add) >= needed:
                        break

        cands.extend(popular_to_add)

    # ГАРАНТИРУЕМ что вернем ровно max_candidates
    return cands[:max_candidates]


# ===== ФУНКЦИЯ ДЛЯ РАСЧЕТА УНИКАЛЬНОСТИ =====
def calculate_uniqueness(recommendations):
    """Вычисляет процент уникальных рекомендаций"""
    all_items = []
    for items in recommendations.values():
        all_items.extend(items)

    total_recommendations = len(all_items)
    unique_recommendations = len(set(all_items))

    return (
        (unique_recommendations / total_recommendations * 100)
        if total_recommendations > 0
        else 0
    )


def ensure_top_k_recommendations(
    recommendations_dict, top_k=100, popular_items_array=None
):
    """Гарантирует что у каждого пользователя ровно top_k рекомендаций"""
    if popular_items_array is None:
        popular_items_array = np.array([], dtype=np.int64)

    fixed_recommendations = {}
    problematic_users = 0

    for user_id, items in recommendations_dict.items():
        if len(items) < top_k:
            problematic_users += 1
            # Добираем популярными
            needed = top_k - len(items)
            additional = popular_items_array[:needed].tolist()
            fixed_items = items + additional
            fixed_recommendations[user_id] = fixed_items[:top_k]
        elif len(items) > top_k:
            fixed_recommendations[user_id] = items[:top_k]
        else:
            fixed_recommendations[user_id] = items

    if problematic_users > 0:
        log_message(f"Исправлено рекомендаций для {problematic_users} пользователей")

    return fixed_recommendations


# ===== ОСНОВНОЙ ИНФЕРЕНС =====
def main():
    start_time = time.time()
    log_message("=== Старт инференса XGBoost ===")

    # --- Загрузка пользователей
    log_message("Загрузка пользователей...")
    parquet_files = glob.glob(TEST_USERS_PATH)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found matching: {TEST_USERS_PATH}")

    test_dfs = []
    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        test_dfs.append(df)
        log_message(f"Загружен файл: {file_path}, users: {len(df)}")

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

    if hasattr(model, "feature_names") and model.feature_names:
        if model.feature_names != feature_columns:
            log_message("Выравнивание фичей модели и артефактов")
            feature_columns = model.feature_names

    item_map_set = set(artifacts["item_map"].keys())
    popular_items_array = np.array(artifacts["popular_items"], dtype=np.int64)

    log_message(f"Загружено {len(feature_columns)} признаков")
    log_message(f"User features: {len(artifacts['user_features_dict'])}")
    log_message(f"Item features: {len(artifacts['item_features_dict'])}")

    # --- Генерация рекомендаций батчами
    recommendations = {}
    header_written = False
    processed = 0

    log_message("Начинаем генерацию кандидатов...")

    # ПРЕДВАРИТЕЛЬНАЯ генерация всех кандидатов
    all_candidates_list = []
    all_user_ids = []

    with tqdm(total=len(test_users), desc="Генерация кандидатов") as pbar:
        for i in range(0, len(test_users), BATCH_USERS):
            batch_users = test_users[i : i + BATCH_USERS]
            batch_candidates = []
            batch_user_ids = []

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
                        max_candidates=300,
                        artifacts=artifacts,
                    )
                    batch_candidates.append(cands)
                    batch_user_ids.append(uid)
                except Exception as e:
                    log_message(f"Ошибка генерации кандидатов для user {uid}: {e}")
                    # Fallback
                    start_idx = (uid % 50) * 2
                    fallback = popular_items_array[start_idx : start_idx + 300].tolist()
                    batch_candidates.append(fallback)
                    batch_user_ids.append(uid)

            all_candidates_list.extend(batch_candidates)
            all_user_ids.extend(batch_user_ids)
            pbar.update(len(batch_users))

    log_message("Генерация кандидатов завершена, начинаем ранжирование...")

    # --- ОСНОВНОЙ ЦИКЛ РАНЖИРОВАНИЯ С ПРАВИЛЬНОЙ ОБРАБОТКОЙ ОШИБОК
    with tqdm(total=len(test_users), desc="Ранжирование") as pbar:
        for i in range(0, len(test_users), BATCH_USERS):
            batch_end = min(i + BATCH_USERS, len(test_users))
            batch_user_ids = all_user_ids[i:batch_end]
            batch_candidates = all_candidates_list[i:batch_end]

            batch_recommendations = {}

            try:
                # ДЕБАГ: проверяем что батч не пустой
                if not batch_user_ids:
                    log_message(f"Пустой батч на индексе {i}")
                    continue

                log_message(
                    f"Обработка батча {i}-{batch_end}, users: {len(batch_user_ids)}"
                )

                X_batch = make_candidate_frame_batch(
                    user_ids=batch_user_ids,
                    all_candidates=batch_candidates,
                    user_features_dict=artifacts["user_features_dict"],
                    item_features_dict=artifacts["item_features_dict"],
                    feature_columns=feature_columns,
                    cat_features=cat_features,
                    item_to_cat=artifacts.get("item_to_cat"),
                    recent_items_map=artifacts.get("recent_items_map"),
                )

                if X_batch.empty:
                    log_message(f"Пустой DataFrame для батча {i}")
                    # Fallback для всего батча
                    for uid in batch_user_ids:
                        start_idx = (uid % 50) * 2
                        fallback_items = popular_items_array[
                            start_idx : start_idx + TOP_K
                        ].tolist()
                        if len(fallback_items) < TOP_K:
                            additional = popular_items_array[
                                TOP_K : TOP_K + (TOP_K - len(fallback_items))
                            ]
                            fallback_items.extend(additional.tolist())
                        batch_recommendations[uid] = fallback_items[:TOP_K]
                else:
                    log_message(f"Размер батча: {len(X_batch)} строк")

                    # Проверяем наличие всех фичей
                    missing_features = [
                        f for f in feature_columns if f not in X_batch.columns
                    ]
                    if missing_features:
                        log_message(f"Отсутствующие фичи: {missing_features[:5]}...")
                        for f in missing_features:
                            X_batch[f] = 0.0

                    X_pred = X_batch[feature_columns].fillna(0)
                    X_pred = X_pred.replace([np.inf, -np.inf], 0)

                    dmatrix = xgb.DMatrix(X_pred)
                    probs = model.predict(dmatrix)

                    result_df = pd.DataFrame(
                        {
                            "user_id": X_batch["user_id"].values,
                            "item_id": X_batch["item_id"].values,
                            "score": probs,
                        }
                    )

                    log_message(f"Предсказания готовы, размер: {len(result_df)}")

                    # Обрабатываем каждого пользователя
                    for user_id, group in result_df.groupby("user_id"):
                        if len(group) < TOP_K:
                            log_message(
                                f"Мало кандидатов для user {user_id}: {len(group)}"
                            )

                        top_items = group.nlargest(TOP_K * 2, "score")[
                            "item_id"
                        ].tolist()

                        # Гарантируем ровно TOP_K
                        if len(top_items) >= TOP_K:
                            final_items = top_items[:TOP_K]
                        else:
                            # Добираем популярными
                            needed = TOP_K - len(top_items)
                            popular_fallback = []
                            start_idx = (user_id % 100) * 2

                            for j in range(start_idx, len(popular_items_array)):
                                candidate = popular_items_array[j]
                                if candidate not in top_items:
                                    popular_fallback.append(candidate)
                                    if len(popular_fallback) >= needed:
                                        break

                            final_items = top_items + popular_fallback[:needed]

                        # Финальная гарантия
                        if len(final_items) < TOP_K:
                            additional_needed = TOP_K - len(final_items)
                            emergency_items = popular_items_array[
                                :additional_needed
                            ].tolist()
                            final_items.extend(emergency_items)

                        batch_recommendations[user_id] = final_items[:TOP_K]

            except Exception as e:
                log_message(f"КРИТИЧЕСКАЯ ОШИБКА в батче {i}: {e}")
                import traceback

                log_message(f"Трассировка: {traceback.format_exc()}")

                # Fallback для всего батча
                for uid in batch_user_ids:
                    start_idx = (uid % 50) * 2
                    fallback_items = popular_items_array[
                        start_idx : start_idx + TOP_K
                    ].tolist()
                    if len(fallback_items) < TOP_K:
                        additional = popular_items_array[
                            TOP_K : TOP_K + (TOP_K - len(fallback_items))
                        ]
                        fallback_items.extend(additional.tolist())
                    batch_recommendations[uid] = fallback_items[:TOP_K]

            # Добавляем рекомендации батча
            recommendations.update(batch_recommendations)
            processed += len(batch_user_ids)
            pbar.update(len(batch_user_ids))

            # Сохранение каждые 5000 пользователей
            if processed % 5000 == 0:
                if recommendations:
                    # Гарантируем что у всех ровно TOP_K
                    recommendations = ensure_top_k_recommendations(
                        recommendations, TOP_K, popular_items_array
                    )

                    save_recommendations_to_csv(
                        recommendations, OUTPUT_PATH, header=not header_written
                    )
                    header_written = True

                    uniqueness = calculate_uniqueness(recommendations)
                    log_message(
                        f"Обработано {processed}, уникальность: {uniqueness:.1f}%"
                    )

                    recommendations.clear()
                    gc.collect()

    # Финальное сохранение
    if recommendations:
        recommendations = ensure_top_k_recommendations(
            recommendations, TOP_K, popular_items_array
        )
        save_recommendations_to_csv(
            recommendations, OUTPUT_PATH, header=not header_written
        )

    dt = time.time() - start_time
    final_uniqueness = calculate_uniqueness(recommendations)
    log_message(f"Готово! Время: {timedelta(seconds=int(dt))}")
    log_message(f"Скорость: {len(test_users)/dt:.1f} users/sec")
    log_message(f"Финальная уникальность: {final_uniqueness:.1f}%")


if __name__ == "__main__":
    main()

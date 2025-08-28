import gc
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timedelta

import dask.dataframe as dd
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- ЛОГИРОВАНИЕ ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODEL_PATH = "/home/root6/python/e_cup/rec_system/src/models/lgbm_model_full.pkl"
TEST_USERS_PATH = "/home/root6/python/e_cup/rec_system/data/raw/test_users/*.parquet"
OUTPUT_PATH = "/home/root6/python/e_cup/rec_system/result/submission.csv"
TOP_K = 100
USER_BATCH_SIZE = 1000
CANDIDATES_PER_USER = 2000

# --- загрузка модели ---
logger.info("Загружаем модель из %s", MODEL_PATH)
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

# Сначала посмотрим, что на самом деле в model_data
logger.info(f"Тип model_data: {type(model_data)}")

if isinstance(model_data, dict):
    logger.info(f"Ключи в model_data: {list(model_data.keys())}")
    for key, value in model_data.items():
        logger.info(f"model_data['{key}'] тип: {type(value)}")

    # Попробуем найти модель и feature_columns
    model = None
    feature_columns = None

    # Ищем модель
    for key, value in model_data.items():
        if hasattr(value, "predict") or "lightgbm" in str(type(value)).lower():
            model = value
            logger.info(f"Найдена модель в ключе: {key}")
            break

    # Ищем feature_columns
    for key, value in model_data.items():
        if (
            isinstance(value, (list, tuple))
            and len(value) > 0
            and isinstance(value[0], str)
        ):
            feature_columns = value
            logger.info(f"Найдены feature_columns в ключе: {key}")
            break

    if model is None:
        # Если не нашли, возможно model_data и есть модель
        if hasattr(model_data, "predict"):
            model = model_data
            logger.info("model_data является моделью")
        else:
            raise ValueError("Не удалось найти модель в model_data")

    if feature_columns is None:
        # Пробуем получить feature_columns из модели или создать список фич
        try:
            if hasattr(model, "feature_name_"):
                feature_columns = model.feature_name_
            elif hasattr(model, "feature_names"):
                feature_columns = model.feature_names
            else:
                # Создаем feature_columns на основе ожидаемых фич
                feature_columns = [
                    "user_count",
                    "user_mean",
                    "user_orders_count",
                    "item_count",
                    "item_orders_count",
                    "item_category",
                ]
                logger.warning(
                    f"Используем дефолтные feature_columns: {feature_columns}"
                )
        except:
            feature_columns = [
                "user_count",
                "user_mean",
                "user_orders_count",
                "item_count",
                "item_orders_count",
                "item_category",
            ]
            logger.warning(f"Используем дефолтные feature_columns: {feature_columns}")

else:
    # model_data не словарь, возможно это сама модель
    if hasattr(model_data, "predict"):
        model = model_data
        logger.info("model_data является моделью")
        # Пробуем получить feature_columns
        try:
            if hasattr(model, "feature_name_"):
                feature_columns = model.feature_name_
            elif hasattr(model, "feature_names"):
                feature_columns = model.feature_names
            else:
                feature_columns = [
                    "user_count",
                    "user_mean",
                    "user_orders_count",
                    "item_count",
                    "item_orders_count",
                    "item_category",
                ]
                logger.warning(
                    f"Используем дефолтные feature_columns: {feature_columns}"
                )
        except:
            feature_columns = [
                "user_count",
                "user_mean",
                "user_orders_count",
                "item_count",
                "item_orders_count",
                "item_category",
            ]
            logger.warning(f"Используем дефолтные feature_columns: {feature_columns}")
    else:
        raise ValueError(f"Неизвестный тип model_data: {type(model_data)}")

logger.info(f"Тип модели: {type(model)}")
logger.info(f"Количество фич: {len(feature_columns)}")
logger.info(f"Первые 10 фич: {feature_columns[:10]}")


# ===== ВЕКТОРИЗИРОВАННАЯ ФУНКЦИЯ ДЛЯ ПОДГОТОВКИ ФИЧ =====
def prepare_features_vectorized(
    user_id, candidates, user_features_dict, item_features_dict, feature_columns
):
    """Векторизированная подготовка признаков для кандидатов"""
    num_candidates = len(candidates)

    # Создаем матрицу признаков
    features_matrix = np.zeros((num_candidates, len(feature_columns)), dtype=np.float32)

    # Получаем user features
    user_feats = user_features_dict.get(user_id, {})

    # Заполняем user features
    for j, feat_name in enumerate(feature_columns):
        if feat_name in user_feats:
            features_matrix[:, j] = user_feats[feat_name]

    # Заполняем item features
    for j, feat_name in enumerate(feature_columns):
        if any(
            feat_name.startswith(prefix) for prefix in ["item_", "fclip_", "category_"]
        ):
            for i, item_id in enumerate(candidates):
                item_feats = item_features_dict.get(item_id, {})
                if feat_name in item_feats:
                    features_matrix[i, j] = item_feats[feat_name]

    return features_matrix


# ===== ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ =====
def get_user_recommendations(user_id, top_k=100, **kwargs):
    """СУПЕР-БЫСТРАЯ функция: персонализированные рекомендации с векторизацией"""
    try:
        recent_items_get = kwargs.get("recent_items_get")
        popular_items_array = kwargs.get("popular_items_array")
        model = kwargs.get("model")
        feature_columns = kwargs.get("feature_columns")
        copurchase_map = kwargs.get("copurchase_map")
        item_to_cat = kwargs.get("item_to_cat")
        cat_to_items = kwargs.get("cat_to_items")
        item_map = kwargs.get("item_map")
        user_features_dict = kwargs.get("user_features_dict")
        item_features_dict = kwargs.get("item_features_dict")

        # недавние товары пользователя
        recent_items = recent_items_get(user_id, [])

        # Генерация кандидатов
        candidates = set()

        N_RECENT = 5
        N_COPURCHASE = 5
        N_CATEGORY = 5
        N_POPULAR = 30

        # 1. Недавние товары
        candidates.update(recent_items[:N_RECENT])

        # 2. Co-purchase товары
        for item in recent_items[:10]:
            co_items = copurchase_map.get(item, [])
            candidates.update(co_items[:N_COPURCHASE])

        # 3. Товары из тех же категорий
        for item in recent_items[:5]:
            cat_id = item_to_cat.get(item)
            if cat_id and cat_id in cat_to_items:
                candidates.update(cat_to_items[cat_id][:N_CATEGORY])

        # 4. Популярные товары как fallback
        candidates.update(popular_items_array[:N_POPULAR])

        # Фильтруем существующие товары
        candidates = [c for c in candidates if c in item_map]

        if not candidates:
            return popular_items_array[:top_k].tolist()

        # === 🔥 Адаптивный cut-off кандидатов ===
        if len(recent_items) < 3:
            max_cands = 500  # для "новых" пользователей
        else:
            max_cands = 300  # для активных пользователей

        if len(candidates) > max_cands:
            # оставляем наиболее популярные среди кандидатов
            popularity_rank = {
                item: idx for idx, item in enumerate(popular_items_array)
            }
            candidates = sorted(candidates, key=lambda x: popularity_rank.get(x, 1e9))[
                :max_cands
            ]

        # Векторизированная подготовка признаков
        X_candidate = prepare_features_vectorized(
            user_id, candidates, user_features_dict, item_features_dict, feature_columns
        )

        # Предсказания
        predictions = model.predict(X_candidate)

        # Сортируем и получаем топ-K
        sorted_indices = np.argsort(predictions)[::-1][:top_k]
        top_recs = [candidates[i] for i in sorted_indices]

        # Заполняем популярными, если не хватает
        if len(top_recs) < top_k:
            additional_items = []
            for item in popular_items_array:
                if item not in top_recs and item not in additional_items:
                    additional_items.append(item)
                if len(top_recs) + len(additional_items) >= top_k:
                    break
            top_recs.extend(additional_items)
            top_recs = top_recs[:top_k]

        return top_recs

    except Exception as e:
        logger.error(f"Error for user {user_id}: {e}")
        return popular_items_array[:top_k].tolist()


# ===== КЭШ ДЛЯ ПОХОЖИХ ПОЛЬЗОВАТЕЛЕЙ =====
user_recommendation_cache = {}
similar_user_threshold = 5


def get_user_recommendations_with_cache(user_id, top_k=100, **kwargs):
    """Рекомендации с кэшированием"""
    recent_items_get = kwargs.get("recent_items_get")

    if user_id in user_recommendation_cache:
        return user_recommendation_cache[user_id]

    recent_items = recent_items_get(user_id, [])

    for cached_user_id, cached_recs in user_recommendation_cache.items():
        cached_recent = recent_items_get(cached_user_id, [])
        if len(set(recent_items) & set(cached_recent)) >= similar_user_threshold:
            user_recommendation_cache[user_id] = cached_recs
            return cached_recs

    recs = get_user_recommendations(user_id, top_k, **kwargs)
    if len(user_recommendation_cache) < 10000:
        user_recommendation_cache[user_id] = recs

    return recs


# ===== ОБРАБОТКА ВСЕХ ПОЛЬЗОВАТЕЛЕЙ =====
def generate_recommendations_for_users(
    test_users,
    model,
    feature_columns,
    recent_items_map,
    copurchase_map,
    item_to_cat,
    cat_to_items,
    user_features_dict,
    item_features_dict,
    item_map,
    popular_items,
    K,
    log_message,
    output_path=None,
):
    log_message("=== ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ (super_fast + cache) ===")
    stage_start = time.time()

    popular_items_array = np.array(popular_items, dtype=np.int64)
    recent_items_get = recent_items_map.get

    recommendations = {}
    processed = 0
    batch_size = 100  # Увеличиваем батч благодаря оптимизации
    header_written = False

    with tqdm(total=len(test_users), desc="Создание рекомендаций") as pbar:
        for i in range(0, len(test_users), batch_size):
            batch_users = test_users[i : i + batch_size]

            for user_id in batch_users:
                try:
                    recommendations[user_id] = get_user_recommendations_with_cache(
                        user_id,
                        K,
                        recent_items_get=recent_items_get,
                        popular_items_array=popular_items_array,
                        model=model,
                        feature_columns=feature_columns,
                        copurchase_map=copurchase_map,
                        item_to_cat=item_to_cat,
                        cat_to_items=cat_to_items,
                        item_map=item_map,
                        user_features_dict=user_features_dict,
                        item_features_dict=item_features_dict,
                    )
                except Exception as e:
                    recommendations[user_id] = popular_items_array[:K].tolist()
                    log_message(f"Ошибка для пользователя {user_id}: {e}")

                processed += 1
                pbar.update(1)

                if output_path and processed % 10000 == 0:
                    save_recommendations_to_csv(
                        recommendations,
                        output_path,
                        log_message,
                        header=not header_written,
                    )
                    header_written = True
                    recommendations.clear()
                    gc.collect()

    if output_path and recommendations:
        save_recommendations_to_csv(
            recommendations,
            output_path,
            log_message,
            header=not header_written,
        )

    stage_time = time.time() - stage_start
    log_message(f"Генерация завершена за {timedelta(seconds=stage_time)}")

    return recommendations


def save_recommendations_to_csv(recommendations, output_path, log_message, header=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mode = "w" if header else "a"
    with open(output_path, mode, encoding="utf-8", buffering=16384) as f:
        if header:
            f.write("user_id,item_id_1 item_id_2 ... item_id_100\n")
        for user_id, items in recommendations.items():
            items_str = " ".join(str(int(item)) for item in items)
            f.write(f"{int(user_id)},{items_str}\n")


if __name__ == "__main__":
    start_time = time.time()
    log_file = "/home/root6/python/e_cup/rec_system/predict_log.txt"

    def log_message(message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    try:
        log_message("=== Запуск генерации рекомендаций ===")

        # Загрузка данных
        test_df = dd.read_parquet(TEST_USERS_PATH).compute()
        test_users = test_df["user_id"].unique().tolist()
        log_message(f"Загружено {len(test_users)} пользователей")

        # Загрузка вспомогательных данных
        data_paths = {
            "recent_items_map": "/home/root6/python/e_cup/rec_system/data/processed/recent_items_map.pkl",
            "copurchase_map": "/home/root6/python/e_cup/rec_system/data/processed/copurchase_map.pkl",
            "item_to_cat": "/home/root6/python/e_cup/rec_system/data/processed/item_to_cat.pkl",
            "cat_to_items": "/home/root6/python/e_cup/rec_system/data/processed/cat_to_items.pkl",
            "user_features_dict": "/home/root6/python/e_cup/rec_system/data/processed/user_features_dict.pkl",
            "item_features_dict": "/home/root6/python/e_cup/rec_system/data/processed/item_features_dict.pkl",
            "item_map": "/home/root6/python/e_cup/rec_system/data/processed/item_map.pkl",
            "popular_items": "/home/root6/python/e_cup/rec_system/data/processed/popular_items.pkl",
        }

        loaded_data = {}
        for name, path in data_paths.items():
            try:
                with open(path, "rb") as f:
                    loaded_data[name] = pickle.load(f)
                log_message(f"Загружен {name}")
            except Exception as e:
                log_message(f"Ошибка загрузки {name}: {e}")
                raise

        # Генерация рекомендаций
        recommendations = generate_recommendations_for_users(
            test_users=test_users,
            model=model,
            feature_columns=feature_columns,
            recent_items_map=loaded_data["recent_items_map"],
            copurchase_map=loaded_data["copurchase_map"],
            item_to_cat=loaded_data["item_to_cat"],
            cat_to_items=loaded_data["cat_to_items"],
            user_features_dict=loaded_data["user_features_dict"],
            item_features_dict=loaded_data["item_features_dict"],
            item_map=loaded_data["item_map"],
            popular_items=loaded_data["popular_items"],
            K=TOP_K,
            log_message=log_message,
            output_path=OUTPUT_PATH,
        )

        log_message("=== Завершено успешно ===")

    except Exception as e:
        error_msg = f"Ошибка: {str(e)}"
        log_message(error_msg)
        raise

# что сейчас учитывается
# 1) Полный список факторов для формирования рекомендаций:
# 2) ALS рекомендации - коллаборативная фильтрация на основе матричного разложения
# 3) История просмотров пользователя - вес: 2.0 (page_view)
# 4) Добавления в избранное - вес: 5.0 (favorite)
# 5) Добавления в корзину - вес: 10.0 (to_cart)
# 6) Фактор времени - более свежие взаимодействия имеют больший вес
# 7) Глобальная популярность товаров - нормализованный счет популярности
# 8) Последние просмотренные товары - 5 последних items
# 9) Похожие товары - косинусная близость в ALS пространстве
# 10) Товары из той же категории - рекомендации по категориям
# 11) Совместные покупки - товары, которые покупают вместе
# 12) FCLIP эмбеддинги - визуально-текстовые embeddings товаров
# 13) User ALS эмбеддинги - векторные представления пользователей
# 14) Item ALS эмбеддинги - векторные представления товаров
# Ранжирование: Комбинированный скоринг с весами → сортировка по убыванию → топ-100

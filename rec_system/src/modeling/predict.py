import gc
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timedelta

import dask.dataframe as dd
import numpy as np
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

# --- Загрузка модели ---
logger.info(f"Загружаем модель из {MODEL_PATH}")
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

# тумблеры, что используем для рекомендаций. 1 вкл, 0 выкл.
# еще не реализованно. посмотреть какие фичи пришли с моделью, а то всякое может быть....

""""
Активность пользователя
user_count — сколько всего заказов сделал пользователь
user_mean — средняя цена/сумма/др.
user_orders_count — количество уникальных заказов

Характеристики товара
item_count — количество раз, когда товар встречался
item_orders_count — сколько раз товар покупали (популярность!)
item_category — категория товара (числовая кодировка или one-hot)

Совместные признаки
user_item_count — сколько раз пользователь взаимодействовал именно с этим товаром
user_item_recency — давность взаимодействия

Категории
user_category_count — сколько раз юзер покупал из этой категории
user_category_share — доля этой категории у пользователя
item_category_popularity — популярность категории
"""

feature_weights = {
    # Активность пользователя
    "user_count": 1.0,
    "user_mean": 1.0,
    "user_orders_count": 1.0,
    # Характеристики товара
    "item_count": 1.0,
    "item_orders_count": 1.0,
    "item_category": 1.0,
    # Совместные признаки
    "user_item_count": 1.0,
    "user_item_recency": 1.0,
    # Категории
    "user_category_count": 1.0,
    "user_category_share": 1.0,
    "item_category_popularity": 1.0,
}

# --- Поиск модели и feature_columns ---
if isinstance(model_data, dict):
    model = None
    feature_columns = None
    for key, value in model_data.items():
        if hasattr(value, "predict") or "lightgbm" in str(type(value)).lower():
            model = value
        if isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value):
            feature_columns = value

    if model is None:
        raise ValueError("Не удалось найти модель в model_data")
    if feature_columns is None:
        feature_columns = getattr(
            model,
            "feature_name_",
            [
                "user_count",
                "user_mean",
                "user_orders_count",
                "item_count",
                "item_orders_count",
                "item_category",
            ],
        )
else:
    if hasattr(model_data, "predict"):
        model = model_data
        feature_columns = getattr(
            model,
            "feature_name_",
            [
                "user_count",
                "user_mean",
                "user_orders_count",
                "item_count",
                "item_orders_count",
                "item_category",
            ],
        )
    else:
        raise ValueError(f"Неизвестный тип model_data: {type(model_data)}")

logger.info(f"Модель: {type(model)}, количество фич: {len(feature_columns)}")


# ===== УНИВЕРСАЛЬНАЯ ФУНКЦИЯ ВЫРАВНИВАНИЯ ВЕКТОРОВ =====
def resize_vector(vec, target_size):
    vec = np.asarray(vec, dtype=np.float32)
    if len(vec) > target_size:
        return vec[:target_size]
    elif len(vec) < target_size:
        return np.pad(vec, (0, target_size - len(vec)), mode="constant")
    return vec


# ===== ПОДГОТОВКА ФИЧ ДЛЯ КАНДИДАТОВ =====
def prepare_features_vectorized(
    user_feats_array, candidates, item_features_dict, feature_columns
):
    """
    user_feats_array: любая структура user_features_dict[uid]
    candidates: список item_id
    item_features_dict: dict[item_id] -> np.ndarray
    """
    expected_dim = len(feature_columns)
    user_vec = get_user_vector(user_feats_array, feature_columns)
    user_vec = resize_vector(user_vec, expected_dim)

    item_vecs = []
    for item_id in candidates:
        item_vec = item_features_dict.get(item_id)
        if item_vec is None:
            item_vec = np.zeros(expected_dim, dtype=np.float32)
        else:
            item_vec = resize_vector(item_vec, expected_dim)
        final_vec = user_vec + item_vec
        item_vecs.append(final_vec)

    return np.array(item_vecs, dtype=np.float32)


# ===== ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ ДЛЯ ОДНОГО ПОЛЬЗОВАТЕЛЯ =====
def get_user_recommendations(user_id, top_k=100, **kwargs):
    try:
        user_features_dict = kwargs["user_features_dict"]
        item_features_dict = kwargs["item_features_dict"]
        model = kwargs["model"]
        feature_columns = kwargs["feature_columns"]
        recent_items_get = kwargs["recent_items_get"]
        popular_items_array = kwargs["popular_items_array"]
        copurchase_map = kwargs["copurchase_map"]
        item_to_cat = kwargs["item_to_cat"]
        cat_to_items = kwargs["cat_to_items"]
        item_map = kwargs["item_map"]

        recent_items = recent_items_get(user_id, [])

        # Генерация кандидатов
        candidates = set()
        N_RECENT = 15  # последние N_RECENT товаров пользователя, чем больше значение тем больше персонализация (купленные не учитываеются)
        N_COPURCHASE = 15  # что покупалось вместе с товаром, увеличение дает больше связанных товаров (что было в одной корзине)
        N_CATEGORY = 10  # товары из той же категории что были недавно куплены
        N_POPULAR = 10  # просто топ популярынх товаров,

        candidates.update(recent_items[:N_RECENT])
        for item in recent_items[:10]:
            candidates.update(copurchase_map.get(item, [])[:N_COPURCHASE])
        for item in recent_items[:5]:
            cat_id = item_to_cat.get(item)
            if cat_id and cat_id in cat_to_items:
                candidates.update(cat_to_items[cat_id][:N_CATEGORY])
        candidates.update(popular_items_array[:N_POPULAR])
        candidates = [c for c in candidates if c in item_map]

        if not candidates:
            return popular_items_array[:top_k].tolist()

        if len(recent_items) < 3:
            max_cands = 500
        else:
            max_cands = 300
        if len(candidates) > max_cands:
            popularity_rank = {
                item: idx for idx, item in enumerate(popular_items_array)
            }
            candidates = sorted(candidates, key=lambda x: popularity_rank.get(x, 1e9))[
                :max_cands
            ]

        X_candidate = prepare_features_vectorized(
            user_features_dict[user_id], candidates, item_features_dict, feature_columns
        )
        predictions = model.predict(X_candidate)

        sorted_indices = np.argsort(predictions)[::-1][:top_k]
        top_recs = [candidates[i] for i in sorted_indices]

        # Заполнение популярными
        if len(top_recs) < top_k:
            for item in popular_items_array:
                if item not in top_recs:
                    top_recs.append(item)
                if len(top_recs) >= top_k:
                    break

        return top_recs

    except Exception as e:
        # logger.error(f"Error for user {user_id}: {e}")
        return popular_items_array[:top_k].tolist()


# ===== КЭШ =====
user_recommendation_cache = {}
similar_user_threshold = 5


def get_user_recommendations_with_cache(user_id, top_k=100, **kwargs):
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


# ===== ГЕНЕРАЦИЯ ДЛЯ ВСЕХ ПОЛЬЗОВАТЕЛЕЙ =====
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
    stage_start = time.time()
    popular_items_array = np.array(popular_items, dtype=np.int64)
    recent_items_get = recent_items_map.get
    recommendations = {}
    processed = 0
    batch_size = 100
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
            recommendations, output_path, log_message, header=not header_written
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
            f.write(f"{int(user_id)},{' '.join(map(str, map(int, items)))}\n")


def get_user_vector(user_feat_entry, feature_columns):
    """
    Преобразует любую структуру user_features_dict[uid] в numpy-вектор нужной длины
    """
    if isinstance(user_feat_entry, np.ndarray):
        if len(user_feat_entry) == 1 and isinstance(user_feat_entry[0], dict):
            user_dict = user_feat_entry[0]
            return np.array(
                [float(user_dict.get(f, 0.0)) for f in feature_columns],
                dtype=np.float32,
            )
        else:
            return np.asarray(user_feat_entry, dtype=np.float32)
    elif isinstance(user_feat_entry, dict):
        return np.array(
            [float(user_feat_entry.get(f, 0.0)) for f in feature_columns],
            dtype=np.float32,
        )
    elif isinstance(user_feat_entry, (float, int, np.number)):
        return np.full(len(feature_columns), float(user_feat_entry), dtype=np.float32)
    else:
        return np.zeros(len(feature_columns), dtype=np.float32)


# ===== MAIN =====
if __name__ == "__main__":
    start_time = time.time()
    log_file = "/home/root6/python/e_cup/rec_system/predict_log.txt"

    def log_message(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        print(full_msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(full_msg + "\n")

    try:
        log_message("=== Запуск генерации рекомендаций ===")
        test_df = dd.read_parquet(TEST_USERS_PATH).compute()
        test_users = test_df["user_id"].unique().tolist()
        log_message(f"Загружено {len(test_users)} пользователей")

        loaded_data = {
            "recent_items_map": model_data["recent_items_map"],
            "copurchase_map": model_data["copurchase_map"],
            "item_to_cat": model_data["item_to_cat"],
            "user_features_dict": model_data["user_features_dict"],
            "item_features_dict": model_data["item_features_dict"],
            "item_map": model_data["item_map"],
            "popular_items": model_data["popular_items"],
        }

        # Создаем cat_to_items
        cat_to_items = {}
        for item_id, cat_id in loaded_data["item_to_cat"].items():
            cat_to_items.setdefault(cat_id, []).append(item_id)
        loaded_data["cat_to_items"] = cat_to_items

        # Выравнивание размерностей user/item фич
        expected_len = len(feature_columns)
        for uid in loaded_data["user_features_dict"]:
            user_feat_val = loaded_data["user_features_dict"][uid]
            if (
                isinstance(user_feat_val, np.ndarray)
                and len(user_feat_val) == 1
                and isinstance(user_feat_val[0], dict)
            ):
                user_feat_dict = user_feat_val[0]
                loaded_data["user_features_dict"][uid] = resize_vector(
                    [user_feat_dict.get(feat, 0) for feat in feature_columns],
                    expected_len,
                )
            elif isinstance(user_feat_val, np.ndarray):
                loaded_data["user_features_dict"][uid] = resize_vector(
                    user_feat_val, expected_len
                )
            else:
                # на всякий случай словарь
                loaded_data["user_features_dict"][uid] = resize_vector(
                    [user_feat_val.get(feat, 0) for feat in feature_columns],
                    expected_len,
                )
        for iid in loaded_data["item_features_dict"]:
            loaded_data["item_features_dict"][iid] = resize_vector(
                loaded_data["item_features_dict"][iid], expected_len
            )

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
        log_message(f"Ошибка: {e}")
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

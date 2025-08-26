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
TEST_USERS_PATH = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_test_for_participants/test_for_participants/*.parquet"
OUTPUT_PATH = "/home/root6/python/e_cup/rec_system/result/submission.csv"
TOP_K = 100
USER_BATCH_SIZE = 1000
CANDIDATES_PER_USER = 2000

# --- загрузка модели ---
logger.info("Загружаем модель из %s", MODEL_PATH)
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)


# ===== СУПЕР-БЫСТРЫЕ РЕКОМЕНДАЦИИ =====
def get_user_recommendations_super_fast(user_id, top_k=100, **kwargs):
    """СУПЕР-БЫСТРАЯ функция: топ-10 по модели, остальные популярные"""
    try:
        recent_items_get = kwargs.get("recent_items_get")
        popular_items_array = kwargs.get("popular_items_array")
        recommender = kwargs.get("recommender")
        copurchase_map = kwargs.get("copurchase_map")
        item_to_cat = kwargs.get("item_to_cat")
        cat_to_items = kwargs.get("cat_to_items")
        item_map = kwargs.get("item_map")
        user_features_dict = kwargs.get("user_features_dict")
        item_features_dict = kwargs.get("item_features_dict")

        # ключевые признаки (только 5–7)
        key_features = [
            "user_count",
            "user_mean",
            "user_orders_count",  # user
            "item_count",
            "item_orders_count",
            "item_category",  # item
        ]

        # недавние товары
        recent_items = recent_items_get(user_id, [])

        candidates = set()
        candidates.update(recent_items[:15])  # недавние

        # co-purchase (макс 5 товаров, по 8 кандидатов)
        for item in recent_items[:5]:
            co_items = copurchase_map.get(item, [])
            candidates.update(co_items[:8])

        # товары из категорий (макс 3 товара, по 10 кандидатов)
        for item in recent_items[:3]:
            cat_id = item_to_cat.get(item)
            if cat_id and cat_id in cat_to_items:
                candidates.update(cat_to_items[cat_id][:10])

        # фильтруем и ограничиваем 100
        candidates = [c for c in candidates if c in item_map][:100]
        if not candidates:
            return popular_items_array[:top_k].tolist()

        # датафрейм
        candidate_df = pd.DataFrame(
            {"user_id": [user_id] * len(candidates), "item_id": candidates}
        )

        # добавляем только ключевые признаки
        user_feats = user_features_dict.get(user_id, {})
        for feat_name in key_features:
            if feat_name in user_feats:
                candidate_df[feat_name] = user_feats[feat_name]
            elif feat_name.startswith("item_"):
                for item_id in candidates:
                    item_feats = item_features_dict.get(item_id, {})
                    candidate_df.loc[candidate_df["item_id"] == item_id, feat_name] = (
                        item_feats.get(feat_name, 0)
                    )

        candidate_df = candidate_df.fillna(0)

        # все нужные колонки модели
        for col in recommender.feature_columns:
            if col not in candidate_df.columns:
                candidate_df[col] = 0

        X_candidate = candidate_df[recommender.feature_columns]

        # предсказания
        predictions = recommender.model.predict(X_candidate)
        candidate_df["score"] = predictions
        candidate_df = candidate_df.sort_values("score", ascending=False)

        # топ-10 по модели
        top_model_recs = candidate_df["item_id"].head(10).tolist()

        # добавляем популярные до top_k
        final_recs = top_model_recs.copy()
        for item in popular_items_array:
            if item not in final_recs and len(final_recs) < top_k:
                final_recs.append(item)
            if len(final_recs) >= top_k:
                break

        return final_recs[:top_k]

    except Exception:
        return popular_items_array[:top_k].tolist()


# ===== КЭШ ДЛЯ ПОХОЖИХ ПОЛЬЗОВАТЕЛЕЙ =====
user_recommendation_cache = {}
similar_user_threshold = 5  # если >=5 общих recent items → считаем похожими


def get_user_recommendations_with_cache(user_id, top_k=100, **kwargs):
    """Рекомендации с кэшированием"""
    recent_items_get = kwargs.get("recent_items_get")

    # кэш прямого попадания
    if user_id in user_recommendation_cache:
        return user_recommendation_cache[user_id]

    recent_items = recent_items_get(user_id, [])

    # ищем похожих
    for cached_user_id, cached_recs in user_recommendation_cache.items():
        cached_recent = recent_items_get(cached_user_id, [])
        if len(set(recent_items) & set(cached_recent)) >= similar_user_threshold:
            user_recommendation_cache[user_id] = cached_recs
            return cached_recs

    # вычисляем заново
    recs = get_user_recommendations_super_fast(user_id, top_k, **kwargs)
    if len(user_recommendation_cache) < 10000:  # ограничение размера
        user_recommendation_cache[user_id] = recs

    return recs


# ===== ОБРАБОТКА ВСЕХ ПОЛЬЗОВАТЕЛЕЙ =====
def generate_recommendations_for_users(
    test_users,
    recommender,
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
    batch_size = 50
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
                        recommender=recommender,
                        copurchase_map=copurchase_map,
                        item_to_cat=item_to_cat,
                        cat_to_items=cat_to_items,
                        item_map=item_map,
                        user_features_dict=user_features_dict,
                        item_features_dict=item_features_dict,
                    )
                except Exception as e:
                    recommendations[user_id] = popular_items_array[:K]
                    log_message(f"Ошибка для пользователя {user_id}: {e}")

                processed += 1
                pbar.update(1)

                # каждые 10000 пользователей сохраняем и очищаем словарь
                if output_path and processed % 10000 == 0:
                    save_recommendations_to_csv(
                        recommendations,
                        output_path,
                        log_message,
                        header=not header_written,
                    )
                    header_written = True
                    recommendations.clear()

    # сохранить хвост
    if output_path and recommendations:
        save_recommendations_to_csv(
            recommendations,
            output_path,
            log_message,
            header=not header_written,
        )

    stage_time = time.time() - stage_start
    log_message(f"Генерация завершена за {timedelta(seconds=stage_time)}")

    return {}  # так как всё уже сохранено, возвращаем пустое


# ===== СОХРАНЕНИЕ И СТАТИСТИКА =====
def save_recommendations_to_csv(recommendations, output_path, log_message, header=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mode = "w" if header else "a"
    with open(output_path, mode, encoding="utf-8", buffering=16384) as f:
        if header:
            f.write("user_id,item_id\n")
        for user_id, items in recommendations.items():
            items_str = " ".join(str(int(item)) for item in items)
            f.write(f"{int(user_id)},{items_str}\n")
    log_message(
        f"Сохранено {len(recommendations)} пользователей (header={header}) в {output_path}"
    )


def log_final_statistics(recommendations, item_map, start_time, log_message):
    all_items = {i for recs in recommendations.values() for i in recs}
    log_message(f"Пользователей: {len(recommendations)}")
    log_message(f"Уникальных товаров: {len(all_items)}")
    log_message(f"Охват: {len(all_items)/len(item_map)*100:.1f}%")
    total_time = time.time() - start_time
    log_message(f"Общее время: {timedelta(seconds=total_time)}")


if __name__ == "__main__":
    start_time = time.time()
    log_file = "/home/root6/python/e_cup/rec_system/predict_log.txt"

    def log_message(message: str):
        """Логирование в файл и консоль"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    try:
        log_message("=== Запуск генерации рекомендаций ===")

        # === 1. Загружаем тестовых пользователей ===
        log_message("Загрузка тестовых пользователей...")
        test_df = dd.read_parquet(TEST_USERS_PATH).compute()
        test_users = test_df["user_id"].unique().tolist()
        log_message(f"Загружено {len(test_users)} пользователей")

        # === 2. Загружаем вспомогательные данные ===
        log_message("Загрузка маппингов и фичей...")
        # ⚠️ здесь нужно подгрузить заранее подготовленные pickle/parquet с картами
        with open(
            "/home/root6/python/e_cup/rec_system/data/processed/recent_items_map.pkl",
            "rb",
        ) as f:
            recent_items_map = pickle.load(f)
        with open(
            "/home/root6/python/e_cup/rec_system/data/processed/copurchase_map.pkl",
            "rb",
        ) as f:
            copurchase_map = pickle.load(f)
        with open(
            "/home/root6/python/e_cup/rec_system/data/processed/item_to_cat.pkl", "rb"
        ) as f:
            item_to_cat = pickle.load(f)
        with open(
            "/home/root6/python/e_cup/rec_system/data/processed/extended_cat_to_items.pkl",
            "rb",
        ) as f:
            cat_to_items = pickle.load(f)
        with open(
            "/home/root6/python/e_cup/rec_system/data/processed/user_features_dict.pkl",
            "rb",
        ) as f:
            user_features_dict = pickle.load(f)
        with open(
            "/home/root6/python/e_cup/rec_system/data/processed/item_features_dict.pkl",
            "rb",
        ) as f:
            item_features_dict = pickle.load(f)
        with open(
            "/home/root6/python/e_cup/rec_system/data/processed/item_map.pkl", "rb"
        ) as f:
            item_map = pickle.load(f)
        with open(
            "/home/root6/python/e_cup/rec_system/data/processed/popular_items.pkl", "rb"
        ) as f:
            popular_items = pickle.load(f)

        # === 3. Генерация рекомендаций ===
        generate_recommendations_for_users(
            test_users=test_users,
            recommender=model_data,
            recent_items_map=recent_items_map,
            copurchase_map=copurchase_map,
            item_to_cat=item_to_cat,
            cat_to_items=cat_to_items,
            user_features_dict=user_features_dict,
            item_features_dict=item_features_dict,
            item_map=item_map,
            popular_items=popular_items,
            K=TOP_K,
            log_message=log_message,
            output_path=OUTPUT_PATH,  # <--- передаём путь для инкрементального сохранения
        )

        # === 4. Лог финальной статистики ===
        log_final_statistics({}, item_map, start_time, log_message)

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

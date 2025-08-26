import gc
import logging
import os
import pickle
import time
from collections import defaultdict
from datetime import timedelta

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

# -------------------- Настройка логирования --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class FastSubmissionGenerator:
    def __init__(self, model_path):
        self.start_time = time.time()
        logger.info("Загрузка модели и данных...")
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["lgbm_model"]
        self.feature_columns = model_data["feature_columns"]
        self.user_map = model_data["user_map"]
        self.item_map = model_data["item_map"]
        self.inv_item_map = model_data["inv_item_map"]
        self.popular_items = model_data["popular_items"]

        self.user_features_dict = model_data.get("user_features_dict", {})
        self.item_features_dict = model_data.get("item_features_dict", {})
        self.recent_items_map = model_data.get("recent_items_map", {})
        self.copurchase_map = model_data.get("copurchase_map", {})
        self.item_to_cat = model_data.get("item_to_cat", {})
        self.item_embeddings = model_data.get("item_embeddings")

        self.all_items = list(self.item_map.keys())
        self.n_items = len(self.all_items)

        logger.info("Подготовка item features...")
        self.item_features_matrix = self._prepare_item_features_matrix()
        logger.info("Предварительный расчет похожих товаров...")
        self.similar_items_cache = self._precompute_similar_items()
        logger.info("Группировка товаров по категориям...")
        self.category_items_dict = self._precompute_category_items()

        logger.info(
            f"Готово! Пользователей: {len(self.user_map)}, Товаров: {self.n_items}"
        )

    def _prepare_item_features_matrix(self):
        n_features = len(self.feature_columns)
        features_matrix = np.zeros((self.n_items, n_features), dtype=np.float32)
        for idx, item_id in enumerate(self.all_items):
            if item_id in self.item_features_dict:
                item_feats = self.item_features_dict[item_id]
                if isinstance(item_feats, dict):
                    for feat_name, value in item_feats.items():
                        if feat_name in self.feature_columns:
                            features_matrix[
                                idx, self.feature_columns.index(feat_name)
                            ] = float(value)
                elif isinstance(item_feats, (list, np.ndarray)):
                    for feat_idx, value in enumerate(item_feats):
                        if feat_idx < n_features:
                            features_matrix[idx, feat_idx] = float(value)
        return features_matrix

    def _precompute_similar_items(self, top_n=20):
        if self.item_embeddings is None:
            return {}
        logger.info("Вычисление похожих товаров...")
        nn = NearestNeighbors(n_neighbors=top_n + 1, metric="cosine")
        nn.fit(self.item_embeddings)
        similar_items_cache = {}
        for item_id in tqdm(self.all_items[:10000], desc="ALS similar items"):
            if item_id in self.item_map:
                item_idx = self.item_map[item_id]
                distances, indices = nn.kneighbors(
                    self.item_embeddings[item_idx].reshape(1, -1)
                )
                similar_items_cache[item_id] = [
                    self.inv_item_map[idx] for idx in indices[0][1 : top_n + 1]
                ]
        return similar_items_cache

    def _precompute_category_items(self):
        category_items = defaultdict(list)
        for item_id, category in self.item_to_cat.items():
            category_items[category].append(item_id)
        return category_items

    def _get_user_features_vector(self, user_id):
        if user_id in self.user_features_dict:
            user_feats = self.user_features_dict[user_id]
            n_features = len(self.feature_columns)
            user_vector = np.zeros(n_features, dtype=np.float32)
            if isinstance(user_feats, dict):
                for feat_name, value in user_feats.items():
                    if feat_name in self.feature_columns:
                        user_vector[self.feature_columns.index(feat_name)] = float(
                            value
                        )
            elif isinstance(user_feats, (list, np.ndarray)):
                for feat_idx, value in enumerate(user_feats):
                    if feat_idx < n_features:
                        user_vector[feat_idx] = float(value)
            return user_vector
        return None

    def _compute_user_item_scores(self, user_id, top_candidates=1000):
        """Предсказание только для топ-кандидатов (500-1000) с учетом разных типов взаимодействий"""
        candidate_items = set(self.popular_items[:500])  # популярные
        recent_interactions = self.recent_items_map.get(user_id, [])

        # Берем последние 100 взаимодействий
        recent_interactions = recent_interactions[-100:]

        # Добавляем последние просмотры, добавления в избранное и в корзину
        for inter in recent_interactions:
            candidate_items.add(inter["item_id"])
            # похожие, категория, совместные покупки
            candidate_items.update(self.similar_items_cache.get(inter["item_id"], []))
            cat = self.item_to_cat.get(inter["item_id"])
            if cat:
                candidate_items.update(self.category_items_dict.get(cat, []))
            candidate_items.update(self.copurchase_map.get(inter["item_id"], []))

        candidate_items = list(candidate_items)[:top_candidates]
        candidate_indices = [
            self.item_map[i] for i in candidate_items if i in self.item_map
        ]

        if not candidate_indices:
            # cold start
            return self.popular_items[:100]

        batch_features = self.item_features_matrix[candidate_indices].copy()
        user_vec = self._get_user_features_vector(user_id)
        user_feat_indices = [
            i for i, col in enumerate(self.feature_columns) if col.startswith("user_")
        ]
        if user_vec is not None and user_feat_indices:
            batch_features[:, user_feat_indices] = user_vec[user_feat_indices]

        scores = self.model.predict(batch_features, num_iteration=-1)

        # применяем веса для разных типов взаимодействий
        type_weights = {"page_view": 2.0, "favorite": 5.0, "to_cart": 10.0}
        for i, item_id in enumerate(candidate_items):
            for inter in recent_interactions:
                if inter["item_id"] == item_id:
                    scores[i] += type_weights.get(
                        inter.get("action_type", "page_view"), 0.0
                    )
            # базовая популярность
            if item_id in self.popular_items:
                scores[i] += 1.0

        # топ-100
        n_top = min(100, len(scores))
        top_idx = np.argsort(-scores)[:n_top]

        # если меньше 100, дополняем популярными
        recommended = [candidate_items[i] for i in top_idx]
        if len(recommended) < 100:
            recommended += [i for i in self.popular_items if i not in recommended][
                : 100 - len(recommended)
            ]

        return recommended

    def generate_recommendations_batch(
        self, test_users_df, output_path, batch_size=1000
    ):
        user_ids = test_users_df["user_id"].tolist()
        total_users = len(user_ids)
        submission_data = []

        for batch_start in tqdm(range(0, total_users, batch_size), desc="Users"):
            batch_end = min(batch_start + batch_size, total_users)
            batch_user_ids = user_ids[batch_start:batch_end]

            for user_id in batch_user_ids:
                try:
                    recommended_items = self._compute_user_item_scores(user_id)
                    submission_data.append(
                        {
                            "user_id": user_id,
                            "item_id_1 item_id_2 ... item_id_100": " ".join(
                                map(str, recommended_items)
                            ),
                        }
                    )
                except Exception:
                    submission_data.append(
                        {
                            "user_id": user_id,
                            "item_id_1 item_id_2 ... item_id_100": " ".join(
                                map(str, self.popular_items[:100])
                            ),
                        }
                    )

            self._save_submission_batch(submission_data, output_path)
            submission_data = []

    def _save_submission_batch(self, submission_data, output_path):
        df = pd.DataFrame(submission_data)
        file_exists = os.path.exists(output_path)
        df.to_csv(output_path, index=False, mode="a", header=not file_exists)

    def load_test_users(self, test_users_path):
        logger.info("Загрузка тестовых пользователей...")
        test_users_ddf = dd.read_parquet(test_users_path, columns=["user_id"])
        return test_users_ddf.compute()


if __name__ == "__main__":
    start_time = time.time()
    MODEL_PATH = "/home/root6/python/e_cup/rec_system/src/models/lgbm_model_full.pkl"
    TEST_USERS_PATH = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_test_for_participants/test_for_participants/*.parquet"
    OUTPUT_PATH = "/home/root6/python/e_cup/rec_system/result/submission.csv"

    try:
        generator = FastSubmissionGenerator(MODEL_PATH)
        test_users = generator.load_test_users(TEST_USERS_PATH)
        generator.generate_recommendations_batch(
            test_users, OUTPUT_PATH, batch_size=500
        )
    except Exception as e:
        logger.exception(f"Критическая ошибка: {e}")
    finally:
        if os.path.exists(OUTPUT_PATH):
            df = pd.read_csv(OUTPUT_PATH).drop_duplicates(
                subset=["user_id"], keep="first"
            )
            for idx, row in df.iterrows():
                items = row["item_id_1 item_id_2 ... item_id_100"].split()
                if len(items) != 100:
                    df.at[idx, "item_id_1 item_id_2 ... item_id_100"] = " ".join(
                        map(str, generator.popular_items[:100])
                    )
            df.to_csv(OUTPUT_PATH, index=False)
            logger.info(f"✅ Сабмит сохранен: {OUTPUT_PATH}")

    elapsed = time.time() - start_time
    logger.info(f"Общее время: {timedelta(seconds=int(elapsed))}")
    logger.info(f"Скорость: {len(test_users)/elapsed:.1f} users/sec")


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

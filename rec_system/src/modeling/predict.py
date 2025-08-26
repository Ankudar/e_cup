import gc
import glob
import logging
import os
import pickle
import time
from collections import defaultdict
from datetime import timedelta

import dask.dataframe as dd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
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

        # Конфигурация тумблеров
        self.config = {
            "use_popularity": True,
            "use_recent_interactions": True,
            "use_similar_items": True,
            "use_category_items": True,
            "use_copurchase": True,
            "use_als_recommendations": True,
            "use_user_features": True,
            "use_type_weights": True,
            "use_time_decay": True,
        }

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

        # Предвычисленная маска пользовательских признаков
        self.user_feat_mask = np.array(
            [col.startswith("user_") for col in self.feature_columns]
        )

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
        logger.info("Вычисление всех похожих товаров...")
        nn = NearestNeighbors(n_neighbors=top_n + 1, metric="cosine", n_jobs=-1)
        nn.fit(self.item_embeddings)
        distances, indices = nn.kneighbors(self.item_embeddings)

        similar_items_cache = {}
        for item_idx in range(len(self.all_items)):
            item_id = self.inv_item_map[item_idx]
            similar_items_cache[item_id] = [
                self.inv_item_map[idx] for idx in indices[item_idx][1 : top_n + 1]
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

    def _compute_user_item_scores(self, user_id, top_candidates=500):
        candidate_scores = defaultdict(float)

        # Популярные товары
        for i, item_id in enumerate(self.popular_items[:100]):
            candidate_scores[item_id] = 1.0 - (i / 100)

        recent_interactions = self.recent_items_map.get(user_id, [])[-50:]
        for inter in recent_interactions:
            iid = inter["item_id"]
            candidate_scores[iid] += 0
            for sim_item in self.similar_items_cache.get(iid, [])[:5]:
                candidate_scores[sim_item] += 0
            for cat_item in self.category_items_dict.get(
                self.item_to_cat.get(iid, []), []
            )[:5]:
                candidate_scores[cat_item] += 0
            for cp_item in self.copurchase_map.get(iid, [])[:5]:
                candidate_scores[cp_item] += 0

        candidate_items = list(candidate_scores)[:top_candidates]
        if not candidate_items:
            return self.popular_items[:100]

        candidate_indices = [
            self.item_map[i] for i in candidate_items if i in self.item_map
        ]
        batch_features = self.item_features_matrix[candidate_indices].copy()

        if self.config["use_user_features"]:
            user_vec = self._get_user_features_vector(user_id)
            if user_vec is not None:
                batch_features[:, self.user_feat_mask] = user_vec[self.user_feat_mask]

        scores = self.model.predict(batch_features, num_iteration=-1)

        if self.config["use_type_weights"]:
            type_weights = {"page_view": 2.0, "favorite": 5.0, "to_cart": 10.0}
            for i, item_id in enumerate(candidate_items):
                for inter in recent_interactions:
                    if inter["item_id"] == item_id:
                        scores[i] += type_weights.get(
                            inter.get("action_type", "page_view"), 0.0
                        )

        for i, item_id in enumerate(candidate_items):
            if item_id in self.popular_items[:500]:
                popularity_rank = self.popular_items.index(item_id)
                scores[i] += 1.0 * (1 - popularity_rank / 500)

        n_top = min(100, len(scores))
        top_idx = np.argsort(-scores)[:n_top]
        recommended = [candidate_items[i] for i in top_idx]

        if len(recommended) < 100:
            recommended += [i for i in self.popular_items if i not in recommended][
                : 100 - len(recommended)
            ]

        return recommended

    def generate_recommendations_stream(
        self, test_users_path, output_path, batch_size=500, save_every=10000
    ):
        logger.info("Начало генерации рекомендаций...")
        popular_items_str = " ".join(map(str, self.popular_items[:100]))

        parquet_files = glob.glob(test_users_path)
        if not parquet_files:
            raise FileNotFoundError(f"Нет файлов по пути {test_users_path}")
        file_path = parquet_files[0]

        if os.path.exists(output_path):
            os.remove(output_path)

        ddf = dd.read_parquet(file_path, columns=["user_id"])
        delayed_batches = ddf.to_delayed()

        total_processed = 0
        save_buffer = []

        for delayed_df in tqdm(delayed_batches, desc="Партиции Dask"):
            batch_df = delayed_df.compute()
            for user_id in tqdm(batch_df["user_id"], desc="Пользователи", leave=False):
                try:
                    items = self._compute_user_item_scores(user_id)
                    items_str = " ".join(map(str, items))
                except Exception as e:
                    logger.debug(f"Error for user {user_id}: {e}")
                    items_str = popular_items_str
                save_buffer.append((user_id, items_str))
                total_processed += 1

                # Сохраняем каждые save_every пользователей
                if len(save_buffer) >= save_every:
                    with open(output_path, "a") as f:
                        for uid, items_s in save_buffer:
                            f.write(f"{uid},{items_s}\n")
                    save_buffer = []

            del batch_df
            gc.collect()

        # Сохраняем оставшихся пользователей
        if save_buffer:
            with open(output_path, "a") as f:
                for uid, items_s in save_buffer:
                    f.write(f"{uid},{items_s}\n")
            logger.info(f"Сохранено всего {total_processed} пользователей")

        logger.info("Генерация завершена.")

    def update_config(self, new_config):
        self.config.update(new_config)
        logger.info(f"Конфигурация обновлена: {self.config}")


if __name__ == "__main__":
    start_time = time.time()
    MODEL_PATH = "/home/root6/python/e_cup/rec_system/src/models/lgbm_model_full.pkl"
    TEST_USERS_PATH = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_test_for_participants/test_for_participants/*.parquet"
    OUTPUT_PATH = "/home/root6/python/e_cup/rec_system/result/submission.csv"

    try:
        generator = FastSubmissionGenerator(MODEL_PATH)
        generator.generate_recommendations_stream(
            TEST_USERS_PATH, OUTPUT_PATH, batch_size=500
        )
    except Exception as e:
        logger.exception(f"Критическая ошибка: {e}")

    elapsed = time.time() - start_time
    logger.info(f"Общее время: {timedelta(seconds=int(elapsed))}")


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

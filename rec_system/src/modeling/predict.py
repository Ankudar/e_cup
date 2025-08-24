import os
import pickle
import time
from collections import defaultdict
from datetime import timedelta

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


class FastSubmissionGenerator:
    def __init__(self, model_path):
        print("Загрузка модели и данных...")
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["lgbm_model"]
        self.feature_columns = model_data["feature_columns"]
        self.user_map = model_data["user_map"]
        self.item_map = model_data["item_map"]
        self.inv_item_map = model_data["inv_item_map"]
        self.popular_items = model_data["popular_items"]

        # Все дополнительные данные
        self.user_features_dict = model_data.get("user_features_dict", {})
        self.item_features_dict = model_data.get("item_features_dict", {})
        self.recent_items_map = model_data.get("recent_items_map", {})
        self.copurchase_map = model_data.get("copurchase_map", {})
        self.item_to_cat = model_data.get("item_to_cat", {})
        self.item_embeddings = model_data.get("item_embeddings")

        # Предварительные вычисления
        self.all_items = list(self.item_map.keys())
        self.n_items = len(self.all_items)

        # Предварительно вычисляем матрицу item features
        print("Предварительная подготовка item features...")
        self.item_features_matrix = self._prepare_item_features_matrix()

        # Предварительно вычисляем похожие товары для популярных items
        print("Предварительный расчет похожих товаров...")
        self.similar_items_cache = self._precompute_similar_items()

        # Предварительно вычисляем товары по категориям
        print("Предварительная группировка по категориям...")
        self.category_items_dict = self._precompute_category_items()

        print(f"Готово! Пользователей: {len(self.user_map)}, Товаров: {self.n_items}")

    def _prepare_item_features_matrix(self):
        """Предварительно создаем матрицу всех item features"""
        n_features = len(self.feature_columns)
        features_matrix = np.zeros((self.n_items, n_features), dtype=np.float32)

        for idx, item_id in enumerate(self.all_items):
            if item_id in self.item_features_dict:
                item_feats = self.item_features_dict[item_id]
                if isinstance(item_feats, dict):
                    for feat_name, value in item_feats.items():
                        if feat_name in self.feature_columns:
                            feat_idx = self.feature_columns.index(feat_name)
                            features_matrix[idx, feat_idx] = float(value)
                elif isinstance(item_feats, (list, np.ndarray)):
                    for feat_idx, value in enumerate(item_feats):
                        if feat_idx < n_features:
                            features_matrix[idx, feat_idx] = float(value)

        return features_matrix

    def _precompute_similar_items(self, top_n=20):
        """Предварительно вычисляем похожие товары для всех items"""
        if self.item_embeddings is None:
            return {}

        print("Вычисление похожих товаров...")
        # Используем NearestNeighbors для быстрого поиска
        nn = NearestNeighbors(n_neighbors=top_n + 1, metric="cosine")
        nn.fit(self.item_embeddings)

        similar_items_cache = {}
        for item_id in tqdm(self.all_items[:10000]):  # только для популярных товаров
            if item_id in self.item_map:
                item_idx = self.item_map[item_id]
                distances, indices = nn.kneighbors(
                    self.item_embeddings[item_idx].reshape(1, -1)
                )
                # Исключаем сам товар
                similar_indices = indices[0][1 : top_n + 1]
                similar_items_cache[item_id] = [
                    self.inv_item_map[idx] for idx in similar_indices
                ]

        return similar_items_cache

    def _precompute_category_items(self):
        """Предварительно группируем товары по категориям"""
        category_items = defaultdict(list)
        for item_id, category in self.item_to_cat.items():
            category_items[category].append(item_id)
        return category_items

    def _get_user_features_vector(self, user_id):
        """Быстрое получение user features"""
        if user_id in self.user_features_dict:
            user_feats = self.user_features_dict[user_id]
            n_features = len(self.feature_columns)
            user_vector = np.zeros(n_features, dtype=np.float32)

            if isinstance(user_feats, dict):
                for feat_name, value in user_feats.items():
                    if feat_name in self.feature_columns:
                        feat_idx = self.feature_columns.index(feat_name)
                        user_vector[feat_idx] = float(value)
            elif isinstance(user_feats, (list, np.ndarray)):
                for feat_idx, value in enumerate(user_feats):
                    if feat_idx < n_features:
                        user_vector[feat_idx] = float(value)

            return user_vector
        return None

    def _create_batch_features(self, user_ids):
        """Создает features для батча пользователей"""
        n_users = len(user_ids)
        n_features = self.item_features_matrix.shape[1]

        # Создаем большую матрицу для батча
        batch_features = np.zeros(
            (n_users * self.n_items, n_features), dtype=np.float32
        )

        # Копируем item features для всех пользователей
        for i in range(n_users):
            start_idx = i * self.n_items
            end_idx = start_idx + self.n_items
            batch_features[start_idx:end_idx] = self.item_features_matrix

        # Добавляем user features
        for i, user_id in enumerate(user_ids):
            user_vector = self._get_user_features_vector(user_id)
            if user_vector is not None:
                start_idx = i * self.n_items
                end_idx = start_idx + self.n_items

                # Добавляем user features ко всем товарам этого пользователя
                for feat_idx in range(n_features):
                    if self.feature_columns[feat_idx].startswith("user_"):
                        batch_features[start_idx:end_idx, feat_idx] = user_vector[
                            feat_idx
                        ]

        return batch_features

    def _apply_dynamic_boosts(self, base_scores, user_id, user_items_slice):
        """Применяет динамические бонусы к базовым scores"""
        boosted_scores = base_scores.copy()

        if user_id in self.recent_items_map:
            recent_items = self.recent_items_map[user_id]

            # Бонус за последние просмотры (векторизовано)
            for i, item_id in enumerate(self.all_items):
                if item_id in recent_items:
                    recency_idx = recent_items.index(item_id)
                    recency_bonus = (
                        0.5 * (len(recent_items) - recency_idx) / len(recent_items)
                    )
                    boosted_scores[i] += recency_bonus

        return boosted_scores

    def generate_recommendations_batch(
        self, test_users_df, output_path, batch_size=100
    ):
        """Генерация рекомендаций батчами"""
        print("Быстрая генерация рекомендаций...")
        user_ids = test_users_df["user_id"].tolist()
        n_users = len(user_ids)

        submission_data = []

        for batch_start in tqdm(range(0, n_users, batch_size), desc="Обработка батчей"):
            batch_end = min(batch_start + batch_size, n_users)
            batch_user_ids = user_ids[batch_start:batch_end]

            try:
                # 1. Создаем features для всего батча
                batch_features = self._create_batch_features(batch_user_ids)

                # 2. Предсказываем scores для всего батча
                batch_scores = self.model.predict(batch_features, num_iteration=-1)

                # 3. Обрабатываем каждого пользователя в батче
                for i, user_id in enumerate(batch_user_ids):
                    user_scores = batch_scores[
                        i * self.n_items : (i + 1) * self.n_items
                    ]

                    # 4. Применяем динамические бонусы
                    final_scores = self._apply_dynamic_boosts(user_scores, user_id, i)

                    # 5. Берем топ-100
                    top_indices = np.argsort(-final_scores)[:100]
                    recommended_items = [self.all_items[idx] for idx in top_indices]

                    submission_data.append(
                        {
                            "user_id": user_id,
                            "item_id_1 item_id_2 ... item_id_100": " ".join(
                                map(str, recommended_items)
                            ),
                        }
                    )

            except Exception as e:
                print(f"Ошибка в батче {batch_start}: {e}")
                # Fallback для всего батча
                for user_id in batch_user_ids:
                    submission_data.append(
                        {
                            "user_id": user_id,
                            "item_id_1 item_id_2 ... item_id_100": " ".join(
                                map(str, self.popular_items[:100])
                            ),
                        }
                    )

            # Периодическое сохранение
            if len(submission_data) >= 10000:
                self._save_submission_batch(submission_data, output_path)
                submission_data = []

        if submission_data:
            self._save_submission_batch(submission_data, output_path)

    def _save_submission_batch(self, submission_data, output_path):
        df = pd.DataFrame(submission_data)
        file_exists = os.path.exists(output_path)
        df.to_csv(output_path, index=False, mode="a", header=not file_exists)

    def load_test_users(self, test_users_path):
        print("Загрузка тестовых пользователей...")
        test_users_ddf = dd.read_parquet(test_users_path, columns=["user_id"])
        return test_users_ddf.compute()


# -------------------- Основной запуск --------------------
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
        print(f"Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
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
            print(f"✅ Сабмит сохранен: {OUTPUT_PATH}")

    elapsed = time.time() - start_time
    print(f"Общее время: {timedelta(seconds=elapsed)}")
    print(f"Скорость: {len(test_users) / elapsed:.1f} users/sec")

# что сейчас учитывается
# 1) Полный список факторов для формирования рекомендаций:
# 2) ALS рекомендации - коллаборативная фильтрация на основе матричного разложения
# 3) История просмотров пользователя - вес: 2.0 (page_view)
# 4) Добавления в избранное - вес: 4.0 (favorite)
# 5) Добавления в корзину - вес: 6.0 (to_cart)
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

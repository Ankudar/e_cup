import os
import pickle
import time
from collections import defaultdict
from datetime import timedelta

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


class SubmissionGenerator:
    def __init__(self, model_path, use_gpu=True):
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
        self.user_embeddings = model_data.get("user_embeddings")
        self.item_embeddings = model_data.get("item_embeddings")

        # Веса взаимодействий
        self.interaction_weights = {"page_view": 2.0, "favorite": 4.0, "to_cart": 6.0}

        print(f"Модель загружена: {len(self.feature_columns)} признаков")
        print(f"Пользователей: {len(self.user_map)}, Товаров: {len(self.item_map)}")
        print(f"Недавних взаимодействий: {len(self.recent_items_map)}")

    def _get_user_recent_items(self, user_id):
        """Получаем последние товары пользователя с учетом весов"""
        if user_id in self.recent_items_map:
            return self.recent_items_map[user_id]
        return []

    def _get_similar_items(self, item_id, top_n=20):
        """Находим похожие товары по ALS эмбеддингам"""
        if self.item_embeddings is None or item_id not in self.item_map:
            return []

        item_idx = self.item_map[item_id]
        item_embedding = self.item_embeddings[item_idx].reshape(1, -1)

        # Вычисляем косинусную близость со всеми товарами
        similarities = cosine_similarity(item_embedding, self.item_embeddings)[0]

        # Берем топ-N похожих товаров (исключая сам товар)
        similar_indices = np.argsort(-similarities)[1 : top_n + 1]
        return [self.inv_item_map[idx] for idx in similar_indices]

    def _get_category_items(self, user_id, top_n=15):
        """Товары из категорий, которые пользователь просматривал"""
        if user_id not in self.recent_items_map:
            return []

        user_categories = set()
        for item_id in self.recent_items_map[user_id][:10]:  # последние 10 товаров
            if item_id in self.item_to_cat:
                user_categories.add(self.item_to_cat[item_id])

        category_items = []
        for cat in user_categories:
            for item_id, item_cat in self.item_to_cat.items():
                if item_cat == cat and item_id not in category_items:
                    category_items.append(item_id)
                    if len(category_items) >= top_n:
                        break
            if len(category_items) >= top_n:
                break

        return category_items

    def _get_copurchased_items(self, user_id, top_n=10):
        """Товары, которые покупают вместе с просмотренными"""
        if user_id not in self.recent_items_map:
            return []

        copurchased = set()
        for item_id in self.recent_items_map[user_id][:5]:  # последние 5 товаров
            if item_id in self.copurchase_map:
                for copurchased_id in self.copurchase_map[item_id][
                    :3
                ]:  # топ-3 совместных покупок
                    if copurchased_id != item_id:
                        copurchased.add(copurchased_id)
                        if len(copurchased) >= top_n:
                            break

        return list(copurchased)[:top_n]

    def _create_combined_features(self, user_id, item_ids):
        """Создает комбинированные фичи для пользователя и товаров"""
        n_features = len(self.feature_columns)
        n_items = len(item_ids)

        features_matrix = np.zeros((n_items, n_features), dtype=np.float32)

        # Заполняем item features
        for i, item_id in enumerate(item_ids):
            if item_id in self.item_features_dict:
                item_feats = self.item_features_dict[item_id]
                if isinstance(item_feats, dict):
                    for feat_name, value in item_feats.items():
                        if feat_name in self.feature_columns:
                            idx = self.feature_columns.index(feat_name)
                            features_matrix[i, idx] = float(value)
                elif isinstance(item_feats, (list, np.ndarray)):
                    for idx, value in enumerate(item_feats):
                        if idx < n_features:
                            features_matrix[i, idx] = float(value)

        # Добавляем user features
        if user_id in self.user_features_dict:
            user_feats = self.user_features_dict[user_id]
            if isinstance(user_feats, dict):
                for feat_name, value in user_feats.items():
                    if feat_name in self.feature_columns:
                        idx = self.feature_columns.index(feat_name)
                        features_matrix[:, idx] = float(
                            value
                        )  # одинаково для всех товаров
            elif isinstance(user_feats, (list, np.ndarray)):
                for idx, value in enumerate(user_feats):
                    if idx < n_features:
                        features_matrix[:, idx] = float(value)

        return features_matrix

    def generate_personalized_recommendations(self, test_users_df, output_path):
        """Генерация рекомендаций с учетом всех факторов"""
        print("Генерация персонализированных рекомендаций...")
        submission_data = []

        for user_id in tqdm(test_users_df["user_id"], desc="Пользователи"):
            try:
                # 1. Базовые кандидаты из модели
                all_items = list(self.item_map.keys())
                combined_features = self._create_combined_features(user_id, all_items)
                base_scores = self.model.predict(combined_features, num_iteration=-1)

                # 2. Создаем итоговый scoring с весами
                final_scores = base_scores.copy()

                # 3. Добавляем бонусы за различные факторы
                recent_items = self._get_user_recent_items(user_id)

                # Бонус за последние просмотры
                for i, item_id in enumerate(all_items):
                    if item_id in recent_items:
                        # Больший бонус за более свежие взаимодействия
                        recency_bonus = (
                            0.5
                            * (len(recent_items) - recent_items.index(item_id))
                            / len(recent_items)
                        )
                        final_scores[i] += recency_bonus

                # Бонус за похожие товары
                for recent_item in recent_items[:3]:  # для 3 последних товаров
                    similar_items = self._get_similar_items(recent_item)
                    for item_id in similar_items:
                        if item_id in self.item_map:
                            idx = all_items.index(item_id)
                            final_scores[idx] += 0.3  # бонус за похожесть

                # Бонус за товары из тех же категорий
                category_items = self._get_category_items(user_id)
                for item_id in category_items:
                    if item_id in self.item_map:
                        idx = all_items.index(item_id)
                        final_scores[idx] += 0.2

                # Бонус за совместные покупки
                copurchased = self._get_copurchased_items(user_id)
                for item_id in copurchased:
                    if item_id in self.item_map:
                        idx = all_items.index(item_id)
                        final_scores[idx] += 0.4

                # 4. Ранжируем по итоговому scoring
                top_indices = np.argsort(-final_scores)[:100]
                recommended_items = [all_items[idx] for idx in top_indices]

                submission_data.append(
                    {
                        "user_id": user_id,
                        "item_id_1 item_id_2 ... item_id_100": " ".join(
                            map(str, recommended_items)
                        ),
                    }
                )

            except Exception as e:
                print(f"Ошибка для пользователя {user_id}: {e}")
                # Fallback: популярные товары
                submission_data.append(
                    {
                        "user_id": user_id,
                        "item_id_1 item_id_2 ... item_id_100": " ".join(
                            map(str, self.popular_items[:100])
                        ),
                    }
                )

            # Периодическое сохранение
            if len(submission_data) % 1000 == 0:
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
    MODEL_PATH = "/home/root6/python/e_cup/rec_system/src/models/lgbm_model_full.pkl"
    TEST_USERS_PATH = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_test_for_participants/test_for_participants/*.parquet"
    OUTPUT_PATH = "/home/root6/python/e_cup/rec_system/result/submission.csv"

    try:
        generator = SubmissionGenerator(MODEL_PATH)
        test_users = generator.load_test_users(TEST_USERS_PATH)
        generator.generate_personalized_recommendations(test_users, OUTPUT_PATH)
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

    elapsed_time = timedelta(seconds=time.time() - start_time)
    print(f"Время подготовки рекомендаций: {elapsed_time}")

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

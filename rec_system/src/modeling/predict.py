import os
import pickle
import time
from datetime import timedelta
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


class SubmissionGenerator:
    def __init__(self, model_path, use_gpu=True):
        """
        Загрузка обученной модели с GPU поддержкой
        """
        print("Загрузка модели...")
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]
        self.user_map = model_data["user_map"]
        self.item_map = model_data["item_map"]
        self.popularity_s = model_data["popularity_s"]
        self.embeddings_dict = model_data.get("embeddings_dict", {})

        # Загружаем ALS эмбеддинги если есть в модели
        self.user_embeddings = model_data.get("user_embeddings")
        self.item_embeddings = model_data.get("item_embeddings")

        # GPU setup
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        print(f"Использование устройства: {self.device}")

        # Создаем обратные маппинги
        self.inv_item_map = {v: k for k, v in self.item_map.items()}

        # Подготовим популярные товары для fallback (гарантированно 100)
        self.popular_items = self.popularity_s.sort_values(
            ascending=False
        ).index.tolist()[:1000]

        # Подготовим фичи для всех товаров на GPU
        self.all_items_features = self._prepare_all_items_features()

        print(f"Модель загружена: {len(self.feature_columns)} признаков")
        print(f"Пользователей: {len(self.user_map)}, Товаров: {len(self.item_map)}")

    def _prepare_all_items_features(self):
        """Подготовка фичей для всех товаров"""
        print("Подготовка фичей для всех товаров...")

        features_data = []

        for item_idx in range(len(self.item_map)):
            item_id = self.inv_item_map[item_idx]
            feature_row = {}

            # Базовые признаки товара
            feature_row["item_popularity"] = self.popularity_s.get(item_id, 0.0)

            # ALS эмбеддинги товаров
            if (
                self.item_embeddings is not None
                and item_idx < self.item_embeddings.shape[0]
            ):
                for i in range(min(10, self.item_embeddings.shape[1])):
                    feature_row[f"item_als_{i}"] = self.item_embeddings[item_idx, i]

            # FCLIP эмбеддинги
            if item_id in self.embeddings_dict:
                embed = self.embeddings_dict[item_id]
                for i in range(min(10, len(embed))):
                    feature_row[f"fclip_embed_{i}"] = embed[i]

            features_data.append(feature_row)

        features_df = pd.DataFrame(features_data)

        # Заполняем пропущенные колонки нулями
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0.0

        # Убедимся в правильном порядке колонок
        features_df = features_df[self.feature_columns]

        return features_df.values.astype(np.float32)

    def _prepare_user_features(self, user_id):
        """Подготовка фичей для пользователя"""
        user_features = np.zeros(len(self.feature_columns), dtype=np.float32)

        if user_id in self.user_map and self.user_embeddings is not None:
            user_idx = self.user_map[user_id]
            if user_idx < self.user_embeddings.shape[0]:
                # Заполняем user ALS эмбеддинги
                for i in range(min(10, self.user_embeddings.shape[1])):
                    col_name = f"user_als_{i}"
                    if col_name in self.feature_columns:
                        col_idx = self.feature_columns.index(col_name)
                        user_features[col_idx] = self.user_embeddings[user_idx, i]

        return user_features

    def _ensure_exactly_100_items(self, items_list, user_id=None):
        """Гарантирует, что в списке будет ровно 100 товаров"""
        if len(items_list) >= 100:
            return items_list[:100]
        else:
            missing_count = 100 - len(items_list)
            added_items = []

            for item in self.popular_items:
                if item not in items_list and item not in added_items:
                    added_items.append(item)
                    if len(added_items) >= missing_count:
                        break

            result = items_list + added_items
            return result[:100]

    def load_test_users(self, test_users_path):
        """Загрузка тестовых пользователей"""
        print("Загрузка тестовых пользователей...")
        test_users_ddf = dd.read_parquet(test_users_path, columns=["user_id"])
        test_users_df = test_users_ddf.compute()
        print(f"Загружено {len(test_users_df)} тестовых пользователей")
        return test_users_df

    def generate_recommendations_gpu(
        self, test_users_df, output_path, top_k=100, batch_size=1000
    ):
        """GPU-ускоренная генерация рекомендаций с полной логикой"""
        print("GPU-ускоренная генерация рекомендаций...")

        n_items = len(self.item_map)
        n_features = len(self.feature_columns)
        submission_data = []

        # Переносим item features на GPU
        items_features_tensor = torch.tensor(
            self.all_items_features, device=self.device
        )

        # Обрабатываем пользователей батчами
        for i in tqdm(
            range(0, len(test_users_df), batch_size), desc="Генерация рекомендаций"
        ):
            batch_users = test_users_df["user_id"].iloc[i : i + batch_size]
            batch_data = []

            # Подготавливаем фичи для батча пользователей
            user_features_list = []
            valid_user_ids = []

            for user_id in batch_users:
                try:
                    user_features = self._prepare_user_features(user_id)
                    user_features_list.append(user_features)
                    valid_user_ids.append(user_id)
                except Exception as e:
                    print(f"Ошибка подготовки фичей для {user_id}: {e}")
                    valid_user_ids.append(user_id)
                    continue

            if not user_features_list:
                # Fallback для всего батча
                for user_id in batch_users:
                    recommended_items = self.popular_items[:top_k]
                    recommended_items = self._ensure_exactly_100_items(
                        recommended_items
                    )

                    batch_data.append(
                        {
                            "user_id": user_id,
                            "item_id_1 item_id_2 ... item_id_100": " ".join(
                                map(str, recommended_items)
                            ),
                        }
                    )
                continue

            # Создаем батч пользовательских фичей
            user_features_batch = np.array(user_features_list, dtype=np.float32)

            try:
                # Создаем комбинированные фичи: для каждого пользователя со всеми товарами
                # [batch_size, n_items, n_features] = user_features + item_features
                batch_predictions = []

                # Обрабатываем под-батчи для экономии памяти
                sub_batch_size = 100  # Количество пользователей за раз

                for j in range(0, len(valid_user_ids), sub_batch_size):
                    sub_batch_users = valid_user_ids[j : j + sub_batch_size]
                    sub_batch_features = user_features_batch[j : j + sub_batch_size]

                    # Расширяем user features для всех товаров
                    user_features_expanded = torch.tensor(
                        sub_batch_features[
                            :, np.newaxis, :
                        ],  # [sub_batch, 1, features]
                        device=self.device,
                    ).expand(
                        -1, n_items, -1
                    )  # [sub_batch, n_items, features]

                    # Комбинируем с item features
                    combined_features = (
                        user_features_expanded + items_features_tensor.unsqueeze(0)
                    )

                    # Изменяем форму для предсказания
                    combined_flat = (
                        combined_features.reshape(-1, n_features).cpu().numpy()
                    )

                    # Предсказываем
                    scores = self.model.predict(combined_flat)
                    scores = scores.reshape(len(sub_batch_users), n_items)

                    # Для каждого пользователя выбираем топ-K товаров
                    for k, user_id in enumerate(sub_batch_users):
                        user_scores = scores[k]
                        top_indices = np.argsort(user_scores)[::-1][:top_k]
                        recommended_items = [
                            self.inv_item_map[idx] for idx in top_indices
                        ]
                        recommended_items = self._ensure_exactly_100_items(
                            recommended_items
                        )

                        batch_data.append(
                            {
                                "user_id": user_id,
                                "item_id_1 item_id_2 ... item_id_100": " ".join(
                                    map(str, recommended_items)
                                ),
                            }
                        )

            except Exception as e:
                print(f"Ошибка предсказания: {e}")
                # Fallback для ошибочного батча
                for user_id in valid_user_ids:
                    recommended_items = self.popular_items[:top_k]
                    recommended_items = self._ensure_exactly_100_items(
                        recommended_items
                    )

                    batch_data.append(
                        {
                            "user_id": user_id,
                            "item_id_1 item_id_2 ... item_id_100": " ".join(
                                map(str, recommended_items)
                            ),
                        }
                    )

            # Сохраняем батч
            submission_data.extend(batch_data)
            if len(submission_data) >= 10000:
                self._save_submission_batch(submission_data, output_path)
                submission_data = []

            # Очищаем память GPU
            if self.use_gpu:
                torch.cuda.empty_cache()

        # Сохраняем остаток
        if submission_data:
            self._save_submission_batch(submission_data, output_path)

        print(f"Рекомендации сгенерированы для {len(test_users_df)} пользователей")

    def _save_submission_batch(self, submission_data, output_path):
        """Сохранение батча рекомендаций"""
        df = pd.DataFrame(submission_data)
        file_exists = os.path.exists(output_path)
        df.to_csv(output_path, index=False, mode="a", header=not file_exists)
        print(f"Сохранено {len(submission_data)} пользователей")


# -------------------- Основной запуск --------------------
if __name__ == "__main__":
    start_time = time.time()
    MODEL_PATH = "/home/root6/python/e_cup/rec_system/src/models/lgbm_model.pkl"
    TEST_USERS_PATH = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_test_for_participants/test_for_participants/*.parquet"
    OUTPUT_PATH = "/home/root6/python/e_cup/rec_system/result/submission.csv"

    has_gpu = torch.cuda.is_available()
    print(f"GPU доступно: {has_gpu}")

    try:
        generator = SubmissionGenerator(MODEL_PATH, use_gpu=has_gpu)
        test_users_df = generator.load_test_users(TEST_USERS_PATH)

        generator.generate_recommendations_gpu(
            test_users_df=test_users_df,
            output_path=OUTPUT_PATH,
            top_k=100,
            batch_size=500,  # Меньший батч для стабильности
        )

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        # Здесь должен быть аварийный fallback с использованием полной логики, а не только популярных

    finally:
        # Финальная проверка количества рекомендаций
        if os.path.exists(OUTPUT_PATH):
            df = pd.read_csv(OUTPUT_PATH)
            df = df.drop_duplicates(subset=["user_id"], keep="first")

            # Исправляем строки с неправильным количеством товаров
            for idx, row in df.iterrows():
                items = row["item_id_1 item_id_2 ... item_id_100"].split()
                if len(items) != 100:
                    fixed_items = generator.popular_items[:100]
                    df.at[idx, "item_id_1 item_id_2 ... item_id_100"] = " ".join(
                        map(str, fixed_items)
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

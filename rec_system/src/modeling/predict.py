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

        # GPU setup
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        print(f"Использование устройства: {self.device}")

        # Создаем обратные маппинги
        self.inv_item_map = {v: k for k, v in self.item_map.items()}

        # Подготовим фичи для всех товаров один раз на GPU
        self.all_items_features_tensor = self._prepare_all_items_features_tensor()

        print(f"Модель загружена: {len(self.feature_columns)} признаков")
        print(f"Пользователей: {len(self.user_map)}, Товаров: {len(self.item_map)}")

    def _prepare_all_items_features_tensor(self):
        """Подготовка фичей для всех товаров в виде тензора на GPU"""
        print("Подготовка фичей для всех товаров на GPU...")
        features = []

        n_items = len(self.item_map)

        for item_idx in range(n_items):
            item_id = self.inv_item_map[item_idx]

            feature_row = {
                "item_popularity": self.popularity_s.get(item_id, 0.0),
            }

            # ALS эмбеддинги товаров если есть
            if hasattr(self, "user_embeddings") and hasattr(self, "item_embeddings"):
                for i in range(min(10, self.item_embeddings.shape[1])):
                    feature_row[f"item_als_{i}"] = self.item_embeddings[item_idx, i]

            # FCLIP эмбеддинги если есть
            if self.embeddings_dict and item_id in self.embeddings_dict:
                embed = self.embeddings_dict[item_id]
                for i in range(min(10, len(embed))):
                    feature_row[f"fclip_embed_{i}"] = embed[i]

            features.append(feature_row)

        features_df = pd.DataFrame(features)

        # Заполняем пропуски
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0.0

        # Убедимся что порядок колонок правильный
        features_df = features_df[self.feature_columns]

        # Конвертируем в тензор и перемещаем на GPU
        features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
        if self.use_gpu:
            features_tensor = features_tensor.to(self.device)

        print(f"Подготовлены фичи для {len(features_df)} товаров на {self.device}")
        return features_tensor

    def _prepare_user_features_tensor(self, user_id):
        """Подготовка фичей для пользователя в виде тензора"""
        user_features = {}

        # ALS эмбеддинги пользователя если есть
        if hasattr(self, "user_embeddings") and user_id in self.user_map:
            user_idx = self.user_map[user_id]
            for i in range(min(10, self.user_embeddings.shape[1])):
                user_features[f"user_als_{i}"] = self.user_embeddings[user_idx, i]
        else:
            # Заполняем нулями если пользователя нет
            for i in range(10):
                user_features[f"user_als_{i}"] = 0.0

        # Создаем тензор в правильном порядке
        user_feature_vec = np.array(
            [user_features.get(col, 0.0) for col in self.feature_columns],
            dtype=np.float32,
        )
        user_tensor = torch.tensor(user_feature_vec, dtype=torch.float32)

        if self.use_gpu:
            user_tensor = user_tensor.to(self.device)

        return user_tensor

    def load_test_users(self, test_users_path):
        """
        Загрузка тестовых пользователей (только user_id)
        """
        print("Загрузка тестовых пользователей...")
        test_users_ddf = dd.read_parquet(test_users_path, columns=["user_id"])
        test_users_df = test_users_ddf.compute()
        print(f"Загружено {len(test_users_df)} тестовых пользователей")
        return test_users_df

    def generate_recommendations_gpu(
        self, test_users_df, output_path, top_k=100, batch_size=2000
    ):
        """
        GPU-ускоренная генерация рекомендаций с PyTorch
        """
        print("GPU-ускоренная генерация рекомендаций с PyTorch...")
        print(f"Размер батча: {batch_size}, Устройство: {self.device}")

        # Получаем популярные товары для fallback
        popular_items = self.popularity_s.sort_values(ascending=False).index.tolist()[
            :top_k
        ]

        n_items = len(self.item_map)
        submission_data = []

        # Обрабатываем пользователей батчами
        for i in tqdm(range(0, len(test_users_df), batch_size), desc="GPU обработка"):
            batch_users = test_users_df["user_id"].iloc[i : i + batch_size]
            batch_data = []

            # Подготавливаем фичи для батча пользователей
            user_features_list = []
            valid_user_ids = []

            for user_id in batch_users:
                try:
                    user_tensor = self._prepare_user_features_tensor(user_id)
                    user_features_list.append(user_tensor)
                    valid_user_ids.append(user_id)
                except:
                    valid_user_ids.append(user_id)
                    continue

            if not user_features_list:
                # Fallback для всего батча
                for user_id in batch_users:
                    batch_data.append(
                        {
                            "user_id": user_id,
                            "item_id_1 item_id_2 ... item_id_100": " ".join(
                                map(str, popular_items)
                            ),
                        }
                    )
                continue

            # Создаем батч тензоров
            user_features_batch = torch.stack(
                user_features_list
            )  # [batch_size, n_features]

            # Расширяем для всех товаров: [batch_size, n_items, n_features]
            user_features_expanded = user_features_batch.unsqueeze(1).expand(
                -1, n_items, -1
            )

            # Добавляем item features: user_features + item_features
            # item_features_tensor уже на правильном устройстве
            batch_features = (
                user_features_expanded + self.all_items_features_tensor.unsqueeze(0)
            )

            # Изменяем форму для предсказания: [batch_size * n_items, n_features]
            batch_features_flat = batch_features.reshape(-1, len(self.feature_columns))

            try:
                # Конвертируем в numpy для LightGBM (или используем ONNX если нужно)
                batch_features_np = batch_features_flat.cpu().numpy()

                # Предсказываем
                scores = self.model.predict(batch_features_np)
                scores = scores.reshape(len(valid_user_ids), n_items)

                # Для каждого пользователя выбираем топ-K товаров
                for j, user_id in enumerate(valid_user_ids):
                    user_scores = scores[j]
                    top_indices = np.argsort(user_scores)[::-1][:top_k]
                    recommended_items = [self.inv_item_map[idx] for idx in top_indices]

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
                # Fallback для всего батча
                for user_id in valid_user_ids:
                    batch_data.append(
                        {
                            "user_id": user_id,
                            "item_id_1 item_id_2 ... item_id_100": " ".join(
                                map(str, popular_items)
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
        """
        Сохранение батча рекомендаций
        """
        df = pd.DataFrame(submission_data)

        # Проверяем существует ли файл
        file_exists = os.path.exists(output_path)

        df.to_csv(output_path, index=False, mode="a", header=not file_exists)
        print(f"Сохранено {len(submission_data)} пользователей")

    def generate_simple_recommendations(self, test_users_df, output_path, top_k=100):
        """
        Упрощенная версия - только популярные товары для всех пользователей
        """
        print("Генерация упрощенных рекомендаций (популярные товары)...")

        # Получаем популярные товары
        popular_items = self.popularity_s.sort_values(ascending=False).index.tolist()[
            :top_k
        ]
        popular_items_str = " ".join(map(str, popular_items))

        submission_data = []

        for user_id in tqdm(test_users_df["user_id"], desc="Упрощенные рекомендации"):
            submission_data.append(
                {
                    "user_id": user_id,
                    "item_id_1 item_id_2 ... item_id_100": popular_items_str,
                }
            )

            # Сохраняем каждые 100_000 пользователей
            if len(submission_data) % 50_000 == 0:
                self._save_submission_batch(submission_data, output_path)
                submission_data = []

        # Сохраняем оставшиеся данные
        if submission_data:
            self._save_submission_batch(submission_data, output_path)

        print(
            f"Упрощенные рекомендации сгенерированы для {len(test_users_df)} пользователей"
        )


# -------------------- Основной запуск --------------------
if __name__ == "__main__":
    start_time = time.time()
    # Конфигурация
    MODEL_PATH = "/home/root6/python/e_cup/rec_system/src/models/lgbm_model.pkl"
    TEST_USERS_PATH = "/home/root6/python/e_cup/rec_system/data/raw/ml_ozon_recsys_test_for_participants/test_for_participants/*.parquet"
    OUTPUT_PATH = "/home/root6/python/e_cup/rec_system/result/submission.csv"

    # Проверяем GPU
    has_gpu = torch.cuda.is_available()
    print(f"GPU доступно: {has_gpu}")
    if has_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    try:
        # Создаем генератор с GPU
        generator = SubmissionGenerator(MODEL_PATH, use_gpu=has_gpu)

        # Загружаем тестовых пользователей
        test_users_df = generator.load_test_users(TEST_USERS_PATH)

        # Генерируем рекомендации на GPU
        generator.generate_recommendations_gpu(
            test_users_df=test_users_df,
            output_path=OUTPUT_PATH,
            top_k=100,
            batch_size=2000,  # Большой батч для RTX 4090
        )

    except Exception as e:
        print(f"Ошибка при GPU генерации: {e}")
        print("Используем упрощенный метод...")

        # Fallback
        test_users_df = dd.read_parquet(TEST_USERS_PATH, columns=["user_id"]).compute()
        generator.generate_simple_recommendations(test_users_df, OUTPUT_PATH)

    finally:
        # Удаляем дубликаты user_id в финальном файле
        print("Удаление дубликатов в сабмите...")
        if os.path.exists(OUTPUT_PATH):
            df = pd.read_csv(OUTPUT_PATH)
            df = df.drop_duplicates(
                subset=["user_id"], keep="first"
            )  # Удаляем дубликаты, оставляем последнюю запись
            df.to_csv(OUTPUT_PATH, index=False)
            print(
                f"Удалено {len(df) - len(df.drop_duplicates(subset=['user_id']))} дубликатов"
            )

    print(f"✅ Сабмит сохранен: {OUTPUT_PATH}")
    elapsed_time = timedelta(seconds=time.time() - start_time)
    print(f"Время подготовки рекомендаций: {elapsed_time}")

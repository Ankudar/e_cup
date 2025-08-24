import os
import pickle
import time
from datetime import timedelta

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


class SubmissionGenerator:
    def __init__(self, model_path, use_gpu=True):
        print("Загрузка модели...")
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]
        self.user_map = model_data["user_map"]
        self.item_map = model_data["item_map"]
        self.popularity_s = model_data["popularity_s"]
        self.embeddings_dict = model_data.get("embeddings_dict", {})
        self.user_embeddings = model_data.get("user_embeddings")
        self.item_embeddings = model_data.get("item_embeddings")

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        print(f"Использование устройства: {self.device}")

        self.inv_item_map = {v: k for k, v in self.item_map.items()}
        self.popular_items = self.popularity_s.sort_values(
            ascending=False
        ).index.tolist()[:1000]

        self.all_items_features = self._prepare_all_items_features()
        self.default_user_features = self._create_default_user_features()

        # Индексы user-фичей
        self.user_col_indices = np.array(
            [i for i, col in enumerate(self.feature_columns) if col.startswith("user_")]
        )

        print(f"Модель загружена: {len(self.feature_columns)} признаков")
        print(f"Пользователей: {len(self.user_map)}, Товаров: {len(self.item_map)}")

    def _create_default_user_features(self):
        n_features = len(self.feature_columns)
        default_features = np.zeros(n_features, dtype=np.float32)

        if self.user_embeddings is not None:
            user_als_cols = [
                col for col in self.feature_columns if col.startswith("user_als_")
            ]
            for col in user_als_cols:
                col_idx = self.feature_columns.index(col)
                als_idx = int(col.split("_")[-1])
                if als_idx < self.user_embeddings.shape[1]:
                    default_features[col_idx] = np.mean(
                        self.user_embeddings[:, als_idx]
                    )

        return default_features

    def _prepare_all_items_features(self):
        print("Подготовка фичей для всех товаров...")
        n_items = len(self.item_map)
        feature_dict = {
            col: np.zeros(n_items, dtype=np.float32) for col in self.feature_columns
        }
        item_ids = [self.inv_item_map[idx] for idx in range(n_items)]

        for idx, item_id in enumerate(tqdm(item_ids, desc="Обработка товаров")):
            feature_dict["item_popularity"][idx] = self.popularity_s.get(item_id, 0.0)
            if self.item_embeddings is not None:
                n_als = min(10, self.item_embeddings.shape[1])
                for i in range(n_als):
                    col_name = f"item_als_{i}"
                    if col_name in self.feature_columns:
                        feature_dict[col_name][idx] = self.item_embeddings[idx, i]

            n_fclip = 10
            for i in range(n_fclip):
                col_name = f"fclip_embed_{i}"
                embed = self.embeddings_dict.get(item_id)
                if (
                    embed is not None
                    and i < len(embed)
                    and col_name in self.feature_columns
                ):
                    feature_dict[col_name][idx] = embed[i]

        features_df = pd.DataFrame(feature_dict, index=item_ids)
        features_df = features_df[self.feature_columns]
        tensor_features = torch.tensor(
            features_df.values.astype(np.float16), device=self.device
        )
        return tensor_features

    def _prepare_user_features(self, user_ids):
        n_features = len(self.feature_columns)
        batch_features = np.zeros((len(user_ids), n_features), dtype=np.float32)

        for i, user_id in enumerate(
            tqdm(user_ids, desc="Подготовка фичей пользователей")
        ):
            if user_id in self.user_map and self.user_embeddings is not None:
                user_idx = self.user_map[user_id]
                n_als = min(10, self.user_embeddings.shape[1])
                for j in range(n_als):
                    col_name = f"user_als_{j}"
                    if col_name in self.feature_columns:
                        col_idx = self.feature_columns.index(col_name)
                        batch_features[i, col_idx] = self.user_embeddings[user_idx, j]
            else:
                batch_features[i] = self.default_user_features

        return batch_features

    def _ensure_exactly_100_items(self, items_list):
        if len(items_list) >= 100:
            return items_list[:100]
        missing_count = 100 - len(items_list)
        added_items = [item for item in self.popular_items if item not in items_list][
            :missing_count
        ]
        return items_list + added_items

    def load_test_users(self, test_users_path):
        print("Загрузка тестовых пользователей...")
        test_users_ddf = dd.read_parquet(test_users_path, columns=["user_id"])
        test_users_df = test_users_ddf.compute()
        print(f"Загружено {len(test_users_df)} тестовых пользователей")
        return test_users_df

    def generate_recommendations_gpu(
        self, test_users_df, output_path, top_k=100, batch_size=1000
    ):
        print("GPU-ускоренная генерация рекомендаций...")
        n_items = self.all_items_features.shape[0]
        submission_data = []

        # Весь массив item-фичей
        items_features_np = self.all_items_features.cpu().numpy()

        # Разделяем колонки на user и item
        item_cols = [c for c in self.feature_columns if not c.startswith("user_")]
        user_cols = [c for c in self.feature_columns if c.startswith("user_")]
        items_features_np = pd.DataFrame(
            items_features_np, columns=self.feature_columns
        )[item_cols].values

        # Индексы user-фичей внутри combined_features
        user_col_indices_in_all = np.arange(len(user_cols))

        # Прогресс по батчам
        for i in tqdm(
            range(0, len(test_users_df), batch_size), desc="Батчи пользователей"
        ):
            batch_users = test_users_df["user_id"].iloc[i : i + batch_size]
            user_features_batch = self._prepare_user_features(batch_users.tolist())
            user_features_batch = pd.DataFrame(
                user_features_batch, columns=self.feature_columns
            )[user_cols].values

            for user_idx, user_id in enumerate(
                tqdm(batch_users, desc="Пользователи батча", leave=False)
            ):
                combined_features = np.zeros(
                    (n_items, len(item_cols) + len(user_cols)), dtype=np.float32
                )
                combined_features[:, : len(item_cols)] = items_features_np
                user_vector = user_features_batch[user_idx]  # вектор user-фич
                combined_features[:, len(item_cols) :] = np.tile(
                    user_vector, (n_items, 1)
                )  # дублируем для всех item

                scores = self.model.predict(combined_features, num_iteration=-1)
                top_indices = np.argsort(-scores)[:top_k]
                recommended_items = [self.inv_item_map[idx] for idx in top_indices]
                recommended_items = self._ensure_exactly_100_items(recommended_items)

                submission_data.append(
                    {
                        "user_id": user_id,
                        "item_id_1 item_id_2 ... item_id_100": " ".join(
                            map(str, recommended_items)
                        ),
                    }
                )

            if len(submission_data) >= 5000:
                self._save_submission_batch(submission_data, output_path)
                submission_data = []

        if submission_data:
            self._save_submission_batch(submission_data, output_path)

        print(f"Рекомендации сгенерированы для {len(test_users_df)} пользователей")

    def _save_submission_batch(self, submission_data, output_path):
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

    try:
        generator = SubmissionGenerator(MODEL_PATH)
        test_users_df = generator.load_test_users(TEST_USERS_PATH)
        generator.generate_recommendations_gpu(
            test_users_df, OUTPUT_PATH, top_k=100, batch_size=1000
        )
    except Exception as e:
        print(f"Критическая ошибка: {e}")
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

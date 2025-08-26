import gc
import logging
import os
import pickle
import sys
import time

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

# 1. Проверьте тип объекта
print(f"Тип загруженного объекта: {type(model_data)}")

# 2. Если это словарь (чаще всего)
if isinstance(model_data, dict):
    print("Ключи в словаре модели:")
    for key in model_data.keys():
        print(f"  - {key}")
    
    # Посмотрите содержимое основных ключей
    for key, value in model_data.items():
        print(f"\n--- {key} ---")
        print(f"Тип: {type(value)}")
        
        if hasattr(value, 'shape'):
            print(f"Размер: {value.shape}")
        elif hasattr(value, '__len__'):
            print(f"Длина: {len(value)}")
        
        # Для небольших объектов покажите содержимое
        if key in ['feature_importances_', 'feature_names_', 'best_params_']:
            print(f"Содержимое: {value}")

# 3. Если это модель LightGBM
if hasattr(model_data, 'feature_name_'):
    print(f"\nПризнаки модели: {model_data.feature_name_}")
    print(f"Важность признаков: {model_data.feature_importances_}")

# 4. Детальное исследование структуры
def explore_object(obj, depth=0, max_depth=2):
    if depth > max_depth:
        return
    
    indent = "  " * depth
    if hasattr(obj, '__dict__'):
        print(f"{indent}Объект: {type(obj).__name__}")
        for attr_name in dir(obj):
            if not attr_name.startswith('_'):
                attr_value = getattr(obj, attr_name)
                if not callable(attr_value):
                    print(f"{indent}  {attr_name}: {type(attr_value)}")
                    if depth < max_depth:
                        explore_object(attr_value, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)) and len(obj) > 0:
        print(f"{indent}Коллекция [{type(obj).__name__}] с {len(obj)} элементами")
        if len(obj) > 0:
            explore_object(obj[0], depth + 1, max_depth)
    elif isinstance(obj, dict) and len(obj) > 0:
        print(f"{indent}Словарь с {len(obj)} ключами")
        for key, value in list(obj.items())[:3]:  # первые 3 элемента
            print(f"{indent}  {key}: {type(value)}")
            explore_object(value, depth + 1, max_depth)

# Запустите исследование
explore_object(model_data)

model = model_data["lgbm_model"]
feature_columns = model_data["feature_columns"]
user_features_dict = model_data.get("user_features_dict", {})
item_features_dict = model_data.get("item_features_dict", {})
item_map = model_data["item_map"]

popular_items = model_data.get("popular_items", [])
recent_items_map = model_data.get("recent_items_map", {})
copurchase_map = model_data.get("copurchase_map", {})

all_items = list(item_map.keys())
logger.info("Всего товаров в item_map: %d", len(all_items))

# --- заготовка матрицы признаков товаров ---
logger.info("Строим матрицу признаков для товаров...")
n_features = len(feature_columns)
item_features_matrix = {}
for item_id in all_items:
    feats = np.zeros(n_features, dtype=np.float32)
    for f_name, f_val in item_features_dict.get(item_id, {}).items():
        if f_name in feature_columns:
            feats[feature_columns.index(f_name)] = float(f_val)
    item_features_matrix[item_id] = feats
logger.info("Готово: %d товаров с признаками", len(item_features_matrix))


# --- генератор кандидатов ---
def get_candidates(user_id, top_n=CANDIDATES_PER_USER):
    candidates = set()

    # 1. последние товары юзера
    recent_items = recent_items_map.get(user_id, [])
    candidates.update(recent_items)

    # 2. ковизиты
    for item in recent_items:
        if item in copurchase_map:
            candidates.update(copurchase_map[item])

    # 3. fallback на популярные
    if len(candidates) < top_n:
        candidates.update(popular_items[: top_n - len(candidates)])

    return list(candidates)[:top_n]


# --- чтение тестовых пользователей ---
logger.info("Читаем тестовых пользователей из %s", TEST_USERS_PATH)
ddf = dd.read_parquet(TEST_USERS_PATH, columns=["user_id"])
delayed_batches = ddf.to_delayed()
logger.info("Dask разбил на %d партиций", len(delayed_batches))

if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)
logger.info("Файл %s очищен", OUTPUT_PATH)

csv_header = True

for part_idx, delayed_df in enumerate(tqdm(delayed_batches, desc="Dask partitions")):
    try:
        batch_df = delayed_df.compute()
    except Exception as e:
        logger.error("Ошибка при compute партиции %d: %s", part_idx, str(e))
        continue

    user_ids = batch_df["user_id"].dropna().unique().tolist()
    logger.info("Партиция %d: %d пользователей", part_idx, len(user_ids))

    for start in tqdm(range(0, len(user_ids), USER_BATCH_SIZE), desc="User batches"):
        batch_users = user_ids[start : start + USER_BATCH_SIZE]
        logger.info(
            "Обрабатываем пользователей %d–%d (всего %d)",
            start,
            start + len(batch_users),
            len(user_ids),
        )

        batch_feats = []
        batch_meta = []

        for user_id in batch_users:
            candidates = get_candidates(user_id, top_n=CANDIDATES_PER_USER)
            if not candidates:
                logger.debug("У пользователя %s нет кандидатов", user_id)
                continue

            user_vec = user_features_dict.get(user_id, {})

            for cand in candidates:
                item_id = cand[0] if isinstance(cand, tuple) else cand
                if item_id not in item_features_matrix:
                    continue

                vec = item_features_matrix[item_id].copy()
                for f_name, f_val in user_vec.items():
                    if f_name in feature_columns:
                        vec[feature_columns.index(f_name)] = float(f_val)

                batch_feats.append(vec)
                batch_meta.append((user_id, item_id))

        if not batch_feats:
            logger.warning("Пропускаем батч %d: нет признаков", start)
            continue

        batch_feats = np.array(batch_feats, dtype=np.float32)
        logger.info(
            "Батч готов: %d пар (user,item), shape=%s",
            len(batch_feats),
            batch_feats.shape,
        )

        try:
            scores = model.predict(batch_feats, num_iteration=-1)
        except Exception as e:
            logger.error("Ошибка при предсказании: %s", str(e))
            del batch_feats, batch_meta
            gc.collect()
            continue

        user2items = {}
        for (uid, iid), score in zip(batch_meta, scores):
            user2items.setdefault(uid, []).append((iid, score))

        rows = []
        for uid, pairs in user2items.items():
            top_items = [iid for iid, _ in sorted(pairs, key=lambda x: -x[1])[:TOP_K]]
            rows.append([uid, " ".join(map(str, top_items))])

        try:
            pd.DataFrame(
                rows, columns=["user_id", "item_id_1 item_id_2 ... item_id_100"]
            ).to_csv(OUTPUT_PATH, index=False, mode="a", header=csv_header)
            csv_header = False
        except Exception as e:
            logger.error("Ошибка при записи CSV: %s", str(e))

        logger.info("Батч %d записан: %d пользователей", start, len(rows))

        del batch_feats, batch_meta, rows, user2items, scores
        gc.collect()

#         time.sleep(0.01)  # чуть разгрузим CPU


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

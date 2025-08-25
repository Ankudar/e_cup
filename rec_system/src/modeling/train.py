# def check_gpu_availability():
#     """Проверка доступности GPU"""
#     try:
#         import torch

#         if torch.cuda.is_available():
#             print(f"✅ GPU доступно: {torch.cuda.get_device_name(0)}")
#             print(f"   CUDA capability: {torch.cuda.get_device_capability(0)}")
#             print(
#                 f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
#             )
#             return True
#         else:
#             print("❌ GPU не доступно")
#             return False
#     except ImportError:
#         print("❌ PyTorch не установлен")
#         return False

import time
import numpy as np

rows = np.random.randint(0, 10000, 1000000, dtype=np.int32)
cols = np.random.randint(0, 10000, 1000000, dtype=np.int32)

# Вариант 1 (медленный)
start = time.time()
for _ in range(10):
    result = np.vstack([rows, cols])
print(f"np.vstack: {time.time() - start:.4f}s")

# Вариант 2 (быстрый)
start = time.time()
for _ in range(10):
    result = np.empty((2, len(rows)), dtype=np.int32)
    result[0] = rows
    result[1] = cols
print(f"empty + assign: {time.time() - start:.4f}s")
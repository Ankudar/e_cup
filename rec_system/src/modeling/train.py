# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.device(0))
# print(torch.cuda.get_device_name(0))


# import faiss
# print("FAISS GPU available:", hasattr(faiss, "StandardGpuResources"))
# print("FAISS version:", faiss.__version__)


def check_gpu_availability():
    """Проверка доступности GPU"""
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ GPU доступно: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA capability: {torch.cuda.get_device_capability(0)}")
            print(
                f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
            return True
        else:
            print("❌ GPU не доступно")
            return False
    except ImportError:
        print("❌ PyTorch не установлен")
        return False

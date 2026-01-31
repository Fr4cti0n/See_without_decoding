import torch

def get_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def gpu_info() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{name} ({mem:.1f} GB)"

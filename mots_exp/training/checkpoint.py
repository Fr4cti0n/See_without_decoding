import torch
from pathlib import Path
from typing import Any, Dict

def save_checkpoint(path: str | Path, state: Dict[str, Any]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))

def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    return torch.load(str(path), map_location='cpu')

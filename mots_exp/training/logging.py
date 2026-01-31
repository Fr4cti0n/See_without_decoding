from __future__ import annotations
from typing import Dict, Any
import time

def format_metrics(metrics: Dict[str, Any]) -> str:
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)

class ProgressLogger:
    def __init__(self):
        self.start = time.time()
    def epoch(self, epoch: int, total: int, train: Dict[str, Any], val: Dict[str, Any]):
        elapsed = time.time() - self.start
        print(f"Epoch {epoch}/{total} ({elapsed/60:.1f}m) :: TRAIN {format_metrics(train)} || VAL {format_metrics(val)}")

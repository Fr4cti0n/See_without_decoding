"""Minimal model factory.

Kept intentionally tiny: a dict mapping name -> class. No dynamic loading,
no side effects. Extend by importing a new class and adding one line.
"""
from typing import Dict, Type, List, Any

from .optimized_enhanced.tracker import OptimizedEnhancedOffsetTracker

_MODEL_MAP: Dict[str, Type] = {
    "optimized_enhanced_offset_tracker": OptimizedEnhancedOffsetTracker,
}

def list_models() -> List[str]:
    return list(_MODEL_MAP.keys())

def get_model(name: str, **kwargs: Any):
    cls = _MODEL_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_MAP.keys())}")
    return cls(**kwargs)


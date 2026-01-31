"""
MOTS Training Package
====================

This package contains all training-related functionality for MOTS experiments.

Main modules:
- train_optimized_enhanced_offset_tracker: Main training script with temporal memory
- optimized_enhanced_offset_tracker: Model and loss definitions
"""

from .optimized_enhanced_offset_tracker import OptimizedEnhancedOffsetTracker, OptimizedEnhancedLoss

__all__ = ['OptimizedEnhancedOffsetTracker', 'OptimizedEnhancedLoss']

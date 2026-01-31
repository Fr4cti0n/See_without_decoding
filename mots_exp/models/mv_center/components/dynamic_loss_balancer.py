"""
Dynamic Loss Balancer

Automatically adjusts loss weights during training to achieve target distribution.
Prevents any single loss component from dominating the training signal.
"""

import torch
import torch.nn as nn


class DynamicLossBalancer(nn.Module):
    """
    Dynamically balance multiple loss components to achieve target distribution.
    
    Uses exponential moving average to track loss magnitudes and adjusts weights
    to maintain desired proportions.
    
    Example:
        balancer = DynamicLossBalancer(
            target_ratios={'box': 0.50, 'velocity': 0.35, 'confidence': 0.15},
            momentum=0.9,
            update_freq=10
        )
        
        # During training:
        adjusted_weights = balancer.update(loss_dict)
        total_loss = (loss_dict['box'] * adjusted_weights['box'] +
                      loss_dict['velocity'] * adjusted_weights['velocity'] +
                      loss_dict['confidence'] * adjusted_weights['confidence'])
    """
    
    def __init__(self, target_ratios, initial_weights=None, momentum=0.9, 
                 update_freq=10, min_weight=0.01, max_weight=200.0):
        """
        Initialize dynamic loss balancer.
        
        Args:
            target_ratios: Dict of target loss ratios (must sum to 1.0)
                          e.g., {'box': 0.5, 'velocity': 0.35, 'confidence': 0.15}
            initial_weights: Initial loss weights (default: all 1.0)
            momentum: Momentum for exponential moving average (0.9-0.99)
            update_freq: Update weights every N steps
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
        """
        super().__init__()
        
        # Validate target ratios sum to 1.0
        ratio_sum = sum(target_ratios.values())
        if abs(ratio_sum - 1.0) > 1e-5:
            raise ValueError(f"Target ratios must sum to 1.0, got {ratio_sum}")
        
        self.target_ratios = target_ratios
        self.loss_names = sorted(target_ratios.keys())
        self.momentum = momentum
        self.update_freq = update_freq
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Initialize weights
        if initial_weights is None:
            initial_weights = {name: 1.0 for name in self.loss_names}
        
        self.weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(initial_weights.get(name, 1.0)), 
                             requires_grad=False)
            for name in self.loss_names
        })
        
        # Exponential moving average of raw loss values
        self.register_buffer('loss_ema', torch.zeros(len(self.loss_names)))
        self.register_buffer('step_count', torch.tensor(0))
        
        # Statistics tracking
        self.weight_history = {name: [] for name in self.loss_names}
        self.ratio_history = {name: [] for name in self.loss_names}
    
    def update(self, loss_dict, force_update=False):
        """
        Update loss weights based on current losses.
        
        Args:
            loss_dict: Dict of current raw losses
                      e.g., {'box': 0.5, 'velocity': 0.003, 'confidence': 0.01}
            force_update: Force weight update regardless of update_freq
            
        Returns:
            weights_dict: Current loss weights
        """
        # Extract loss values in consistent order
        loss_values = torch.tensor([
            loss_dict[name].item() if isinstance(loss_dict[name], torch.Tensor) 
            else float(loss_dict[name])
            for name in self.loss_names
        ], device=self.loss_ema.device)
        
        # Update exponential moving average
        if self.step_count == 0:
            # First step: initialize with current values
            self.loss_ema.copy_(loss_values)
        else:
            # EMA update: ema = momentum * ema + (1 - momentum) * current
            self.loss_ema.mul_(self.momentum).add_(
                loss_values, alpha=1.0 - self.momentum
            )
        
        self.step_count += 1
        
        # Update weights at specified frequency
        if force_update or (self.step_count % self.update_freq == 0):
            self._update_weights()
        
        # Return current weights as dict
        return {name: self.weights[name].item() for name in self.loss_names}
    
    def _update_weights(self):
        """
        Update weights to achieve target ratios.
        
        Strategy:
        1. Calculate current weighted losses
        2. Compute current ratios
        3. Adjust weights to move ratios toward targets
        """
        # Get current loss magnitudes (from EMA)
        current_losses = self.loss_ema.clone()
        
        # Get current weights
        current_weights = torch.tensor([
            self.weights[name].item() for name in self.loss_names
        ], device=self.loss_ema.device)
        
        # Calculate weighted losses
        weighted_losses = current_losses * current_weights
        total_weighted = weighted_losses.sum()
        
        # Skip if total is too small (avoid division by zero)
        if total_weighted < 1e-8:
            return
        
        # Calculate current ratios
        current_ratios = weighted_losses / total_weighted
        
        # Calculate target ratios as tensor
        target_ratios = torch.tensor([
            self.target_ratios[name] for name in self.loss_names
        ], device=self.loss_ema.device)
        
        # Compute adjustment factors
        # If current ratio < target: increase weight
        # If current ratio > target: decrease weight
        ratio_errors = target_ratios - current_ratios
        
        # Adjust weights proportionally to error
        # Use more aggressive adjustment for larger errors
        adjustment_rate = 0.5  # Increased from 0.1 for faster convergence
        weight_adjustments = 1.0 + (ratio_errors * adjustment_rate)
        
        # Apply adjustments
        new_weights = current_weights * weight_adjustments
        
        # Clamp to valid range
        new_weights = torch.clamp(new_weights, self.min_weight, self.max_weight)
        
        # Update weight parameters
        for i, name in enumerate(self.loss_names):
            self.weights[name].data = new_weights[i:i+1]
        
        # Track history
        for i, name in enumerate(self.loss_names):
            self.weight_history[name].append(new_weights[i].item())
            self.ratio_history[name].append(current_ratios[i].item())
    
    def get_current_stats(self):
        """
        Get current statistics for monitoring.
        
        Returns:
            stats: Dict with weights, ratios, and targets
        """
        # Get current weighted losses
        current_weights = torch.tensor([
            self.weights[name].item() for name in self.loss_names
        ])
        weighted_losses = self.loss_ema * current_weights
        total = weighted_losses.sum()
        
        if total > 1e-8:
            current_ratios = (weighted_losses / total).tolist()
        else:
            current_ratios = [0.0] * len(self.loss_names)
        
        stats = {}
        for i, name in enumerate(self.loss_names):
            stats[name] = {
                'weight': current_weights[i].item(),
                'current_ratio': current_ratios[i],
                'target_ratio': self.target_ratios[name],
                'raw_loss_ema': self.loss_ema[i].item()
            }
        
        return stats
    
    def print_stats(self):
        """Print current balancing statistics."""
        stats = self.get_current_stats()
        
        print(f"\n📊 Dynamic Loss Balancer Stats (Step {self.step_count}):")
        print(f"{'Loss':<15} {'Weight':<10} {'Current%':<12} {'Target%':<10} {'Raw EMA':<10}")
        print("-" * 65)
        
        for name in self.loss_names:
            s = stats[name]
            print(f"{name:<15} {s['weight']:<10.2f} "
                  f"{s['current_ratio']*100:<12.1f} "
                  f"{s['target_ratio']*100:<10.1f} "
                  f"{s['raw_loss_ema']:<10.6f}")
    
    def reset(self):
        """Reset balancer state."""
        self.loss_ema.zero_()
        self.step_count.zero_()
        for name in self.loss_names:
            self.weights[name].data.fill_(1.0)


def create_memory_tracker_balancer(initial_weights=None):
    """
    Create balancer for memory tracker with recommended target ratios.
    
    Target distribution:
    - Box regression: 50% (position accuracy)
    - Velocity consistency: 35% (temporal coherence)
    - Confidence: 15% (prediction quality)
    
    Args:
        initial_weights: Initial weights dict (default: current training weights)
        
    Returns:
        DynamicLossBalancer instance
    """
    if initial_weights is None:
        initial_weights = {
            'box': 1.0,
            'velocity': 0.5,
            'confidence': 0.1
        }
    
    return DynamicLossBalancer(
        target_ratios={
            'box': 0.50,
            'velocity': 0.35,
            'confidence': 0.15
        },
        initial_weights=initial_weights,
        momentum=0.95,  # Slower adaptation for stability
        update_freq=20,  # Update every 20 steps
        min_weight=0.01,
        max_weight=200.0
    )


if __name__ == '__main__':
    # Test the balancer
    print("🧪 Testing Dynamic Loss Balancer")
    print("=" * 60)
    
    # Create balancer
    balancer = create_memory_tracker_balancer()
    
    # Simulate training with imbalanced losses
    print("\n📊 Simulating 100 training steps with imbalanced losses:")
    print("   Initial: box=0.9, velocity=0.004, confidence=0.05")
    
    for step in range(100):
        # Simulate typical loss values (box dominates)
        loss_dict = {
            'box': torch.tensor(0.9 + 0.05 * torch.randn(1).item()),
            'velocity': torch.tensor(0.004 + 0.001 * torch.randn(1).item()),
            'confidence': torch.tensor(0.05 + 0.01 * torch.randn(1).item())
        }
        
        # Update balancer
        weights = balancer.update(loss_dict)
        
        # Print stats every 20 steps
        if (step + 1) % 20 == 0:
            balancer.print_stats()
    
    print("\n✅ Test complete!")
    print("\nExpected behavior:")
    print("   - Box weight should decrease (currently dominates)")
    print("   - Velocity weight should increase significantly")
    print("   - Confidence weight should increase moderately")
    print("   - Final ratios should approach target: 50:35:15")

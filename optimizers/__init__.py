"""Optimizers used across the project.

This package exists to allow imports like:
    from optimizers.lipschitz_momentum import LipschitzMomentumOptimizer

The actual optimizer implementation is shared from the Chest X-ray package to avoid
code duplication.
"""

from .lipschitz_momentum import LipschitzMomentumOptimizer

__all__ = ["LipschitzMomentumOptimizer"]

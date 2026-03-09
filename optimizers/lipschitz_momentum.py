"""Wrapper module to expose the shared LBM optimizer at the top-level.

This keeps the existing import path `optimizers.lipschitz_momentum` working for
both the Chest X-ray and Retinal OCT training scripts.
"""

from chest_xray.optimizers.lipschitz_momentum import LipschitzMomentumOptimizer

__all__ = ["LipschitzMomentumOptimizer"]

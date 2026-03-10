"""Re-exports LipschitzMomentumOptimizer from chest_xray."""

import sys, os

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chest_xray.optimizers.lipschitz_momentum import LipschitzMomentumOptimizer


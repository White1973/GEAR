"""GEAR utilities.

This package provides a custom reward function entrypoint compatible with
VERL/VAGEN `custom_reward_function`.
"""

from .reward_gear import compute_score

__all__ = ["compute_score"]

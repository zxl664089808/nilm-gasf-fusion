"""Leakage-safe 11-class PLAID appliance representation pipeline."""

from .gramian import angular_field, make_rgb_gramian
from .splits import assign_grouped_splits

__all__ = ["angular_field", "make_rgb_gramian", "assign_grouped_splits"]

"""Evaluation metrics for image generation"""

from .fid import compute_fid, compare_models_fid

__all__ = ['compute_fid', 'compare_models_fid']

"""
Collection of classes and functions to perform data transformation
compatible with scikit-learn objects.
"""

from .remove import RemoveFeaturesByNaN, RemoveCorrelatedFeatures, LimmaFS

__all__ = ['remove']

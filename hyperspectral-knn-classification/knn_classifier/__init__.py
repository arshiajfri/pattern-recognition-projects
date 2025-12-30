"""
Hyperspectral KNN Classification with Dimensionality Reduction.

This module provides the HyperspectralAnalyzer class for classifying
hyperspectral images using KNN with correlation-based band selection.
"""

from .analyzer import HyperspectralAnalyzer
from .config import CONFIG

__all__ = ['HyperspectralAnalyzer', 'CONFIG']

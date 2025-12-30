"""
Hyperspectral Band Correlation Analysis.

This module provides the HyperspectralAnalyzer class for analyzing
band-to-band correlation and performing dimensionality reduction.
"""

from .analyzer import HyperspectralAnalyzer
from .config import CONFIG

__all__ = ['HyperspectralAnalyzer', 'CONFIG']

"""
Spectral Signature Continuum Removal using Convex Hull Method.

This module provides the ContinuumRemoval class for removing continuum
from spectral signatures using convex hull computation.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


class ContinuumRemoval:
    """Class to perform continuum removal on spectral signatures using convex hull method."""
    
    def __init__(self, wavelength, spectral_signature):
        """
        Initialize the ContinuumRemoval object.
        
        Args:
            wavelength (np.array): Array of wavelength values
            spectral_signature (np.array): Array of reflectance values
        """
        self.wavelength = wavelength
        self.spectral_signature = spectral_signature
        self.convex_hull = None
        self.convex_points = None
        self.continuum_removed = None
    
    def compute_convex_hull(self):
        """Compute the convex hull continuum for the spectral signature."""
        # Points for convex hull computation
        points = np.column_stack((self.wavelength, self.spectral_signature))
        hull = ConvexHull(points)
        
        # Extract convex hull points as (x,y) pairs
        self.convex_points = points[hull.vertices]
        
        # 1) Remove duplicate points
        self.convex_points = np.unique(self.convex_points, axis=0)
        
        # 2) Interpolate convex hull to original wavelength grid
        self.convex_hull = np.interp(
            self.wavelength,           # Target wavelength axis
            self.convex_points[:, 0],  # Convex hull wavelength axis
            self.convex_points[:, 1]   # Convex hull reflectance values
        )
        
        return self.convex_hull
    
    def compute_continuum_removed(self):
        """Compute the continuum-removed spectrum."""
        if self.convex_hull is None:
            self.compute_convex_hull()
        
        # 3) Compute Continuum-Removed Spectrum
        self.continuum_removed = self.spectral_signature - self.convex_hull
        
        return self.continuum_removed
    
    def plot_analysis(self, title="Spectral Continuum Removal"):
        """
        Plot the continuum removal results in three subplots.
        
        Args:
            title (str): Title for the first subplot showing the original spectrum
        """
        # Compute continuum removed if not already done
        if self.continuum_removed is None:
            self.compute_continuum_removed()
        
        # Create three subplots in one column
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharey=False)
        
        # --- Plot 1: Original Spectrum ---
        axes[0].plot(self.wavelength, self.spectral_signature, 'b-', 
                     label='Spectral Signature')
        axes[0].set_xlabel("Wavelength (nm)")
        axes[0].set_ylabel("Reflectance")
        axes[0].set_title(title)
        axes[0].set_ylim(0, 1)
        axes[0].grid(True)
        axes[0].legend()
        
        # --- Plot 2: Spectrum + Convex Hull ---
        axes[1].plot(self.wavelength, self.spectral_signature, 'b-', 
                     label='Spectral Signature')
        axes[1].plot(self.convex_points[:, 0], self.convex_points[:, 1],
                     'r-', linewidth=2, label='Convex Hull')
        axes[1].set_xlabel("Wavelength (nm)")
        axes[1].set_ylabel("Reflectance")
        axes[1].set_title("Spectrum with Convex Hull")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True)
        axes[1].legend()
        
        # --- Plot 3: Continuum-Removed ---
        axes[2].plot(self.wavelength, self.continuum_removed, 'g-', 
                     label='Continuum‑Removed')
        axes[2].set_xlabel("Wavelength (nm)")
        axes[2].set_ylabel("Normalized Reflectance")
        axes[2].set_title("Continuum‑Removed Spectrum")
        axes[2].grid(True)
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()


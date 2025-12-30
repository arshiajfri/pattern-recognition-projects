"""
Hyperspectral Image Classifier using Distance Metrics.

This module provides the HyperspectralClassifier class for classifying
hyperspectral images using various distance and similarity metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from spectral.io import envi
import scipy.io
import os


class HyperspectralClassifier:
    """Class to classify hyperspectral images using various distance metrics."""
    
    def __init__(self, image_path, header_path, signatures_path):
        """
        Initialize the HyperspectralClassifier.
        
        Args:
            image_path (str): Path to the hyperspectral image file (.img)
            header_path (str): Path to the ENVI header file (.hdr)
            signatures_path (str): Path to the spectral signatures file (.mat)
        """
        self.image_path = image_path
        self.header_path = header_path
        self.signatures_path = signatures_path
        
        # Data attributes
        self.image = None
        self.data = None
        self.normed_data = None
        self.spectral_sigs = None
        self.labels = None
        self.rows = None
        self.cols = None
        
        # Results
        self.distance_maps = {}
        self.similarity_maps = {}
        
        # Load data on initialization
        self._load_data()
    
    def _load_data(self):
        """Load hyperspectral image and spectral signatures."""
        # Load spectral signatures
        mat_data = scipy.io.loadmat(self.signatures_path)
        self.spectral_sigs = mat_data['M']  # Shape: (num_bands, num_endmembers)
        self.labels = mat_data['cood']
        
        # Load hyperspectral image
        self.image = envi.open(self.header_path, self.image_path)
        img_data = self.image.load()
        self.rows, self.cols = img_data.shape[0], img_data.shape[1]
        num_bands = img_data.shape[2]
        
        # Reshape to (num_pixels, num_bands)
        self.data = img_data.reshape((self.rows * self.cols, num_bands))
        
        # Normalize image bands (min-max normalization)
        data_min = np.min(self.data)
        data_max = np.max(self.data)
        self.normed_data = (self.data - data_min) / (data_max - data_min + 1e-10)
    
    def get_label(self, index):
        """Get label name for a given endmember index."""
        return self.labels[index][0][0]
    
    def display_band(self, band_index=20, cmap='gray'):
        """
        Display a single band of the hyperspectral image.
        
        Args:
            band_index (int): Index of the band to display
            cmap (str): Colormap for visualization
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(self.image[:, :, band_index], cmap=cmap)
        plt.title(f'Band {band_index}')
        plt.colorbar()
        plt.show()
    
    # ==================== Distance Metrics ====================
    
    def compute_euclidean(self):
        """Compute Euclidean distance between pixels and signatures."""
        pixels = self.normed_data  # (num_pixels, num_bands)
        sigs = self.spectral_sigs.T  # (num_endmembers, num_bands)
        distances = cdist(pixels, sigs, metric='euclidean')
        self.distance_maps['Euclidean'] = distances.reshape(self.rows, self.cols, -1)
        return self.distance_maps['Euclidean']
    
    def compute_cosine(self):
        """Compute Cosine distance between pixels and signatures."""
        pixels = self.normed_data
        sigs = self.spectral_sigs.T
        distances = cdist(pixels, sigs, metric='cosine')
        self.distance_maps['Cosine'] = distances.reshape(self.rows, self.cols, -1)
        return self.distance_maps['Cosine']
    
    def compute_cityblock(self):
        """Compute Cityblock (Manhattan) distance between pixels and signatures."""
        pixels = self.normed_data
        sigs = self.spectral_sigs.T
        distances = cdist(pixels, sigs, metric='cityblock')
        self.distance_maps['Cityblock'] = distances.reshape(self.rows, self.cols, -1)
        return self.distance_maps['Cityblock']
    
    def compute_canberra(self):
        """Compute Canberra distance between pixels and signatures."""
        pixels = self.normed_data
        sigs = self.spectral_sigs.T
        distances = cdist(pixels, sigs, metric='canberra')
        self.distance_maps['Canberra'] = distances.reshape(self.rows, self.cols, -1)
        return self.distance_maps['Canberra']
    
    # ==================== Similarity/Error Metrics ====================
    
    def compute_sam(self):
        """Compute Spectral Angle Mapper (SAM) between pixels and signatures."""
        bands = self.normed_data.T  # (num_bands, num_pixels)
        sigs = self.spectral_sigs  # (num_bands, num_endmembers)
        
        # Normalize bands and spectral signatures
        bands_norm = bands / np.linalg.norm(bands, axis=0, keepdims=True)
        sigs_norm = sigs / np.linalg.norm(sigs, axis=0, keepdims=True)
        
        # Compute cosine similarity
        cosine_sim = np.clip(np.dot(bands_norm.T, sigs_norm), -1, 1)
        sam_values = np.arccos(cosine_sim)
        
        self.similarity_maps['SAM'] = sam_values.reshape(self.rows, self.cols, -1)
        return self.similarity_maps['SAM']
    
    def compute_scc(self):
        """Compute Spectral Correlation Coefficient (SCC) between pixels and signatures."""
        bands = self.normed_data.T  # (num_bands, num_pixels)
        sigs = self.spectral_sigs  # (num_bands, num_endmembers)
        
        # Center the data
        bands_centered = bands - np.mean(bands, axis=0, keepdims=True)
        sigs_centered = sigs - np.mean(sigs, axis=0, keepdims=True)
        
        # Compute correlation
        numerator = np.dot(bands_centered.T, sigs_centered)
        bands_std = np.sqrt(np.sum(bands_centered**2, axis=0))
        sigs_std = np.sqrt(np.sum(sigs_centered**2, axis=0))
        denominator = np.outer(bands_std, sigs_std)
        
        scc_values = np.divide(numerator, denominator, 
                               out=np.zeros_like(numerator), where=denominator != 0)
        
        self.similarity_maps['SCC'] = scc_values.reshape(self.rows, self.cols, -1)
        return self.similarity_maps['SCC']
    
    def compute_mse(self):
        """Compute Mean Squared Error (MSE) between pixels and signatures."""
        pixels = self.normed_data[:, np.newaxis, :]  # (num_pixels, 1, num_bands)
        sigs = self.spectral_sigs.T[np.newaxis, :, :]  # (1, num_endmembers, num_bands)
        
        diff = pixels - sigs
        mse_values = np.mean(diff**2, axis=2)
        
        self.similarity_maps['MSE'] = mse_values.reshape(self.rows, self.cols, -1)
        return self.similarity_maps['MSE']
    
    def compute_rmse(self):
        """Compute Root Mean Squared Error (RMSE) between pixels and signatures."""
        if 'MSE' not in self.similarity_maps:
            self.compute_mse()
        
        self.similarity_maps['RMSE'] = np.sqrt(self.similarity_maps['MSE'])
        return self.similarity_maps['RMSE']
    
    # ==================== Compute All ====================
    
    def compute_all_distances(self):
        """Compute all distance metrics."""
        self.compute_euclidean()
        self.compute_cosine()
        self.compute_cityblock()
        self.compute_canberra()
        return self.distance_maps
    
    def compute_all_similarities(self):
        """Compute all similarity/error metrics."""
        self.compute_sam()
        self.compute_scc()
        self.compute_mse()
        self.compute_rmse()
        return self.similarity_maps
    
    def compute_all(self):
        """Compute all distance and similarity metrics."""
        self.compute_all_distances()
        self.compute_all_similarities()
        return {**self.distance_maps, **self.similarity_maps}
    
    # ==================== Visualization ====================
    
    def plot_metric_maps(self, metric_name, figsize=(18, 12), cmap='viridis'):
        """
        Plot distance/similarity maps for a specific metric.
        
        Args:
            metric_name (str): Name of the metric to plot
            figsize (tuple): Figure size
            cmap (str): Colormap for visualization
        """
        all_maps = {**self.distance_maps, **self.similarity_maps}
        
        if metric_name not in all_maps:
            raise ValueError(f"Metric '{metric_name}' not computed. Available: {list(all_maps.keys())}")
        
        metric_map = all_maps[metric_name]
        num_endmembers = metric_map.shape[2]
        
        # Calculate grid dimensions
        ncols = 4
        nrows = (num_endmembers + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()
        
        for i in range(num_endmembers):
            im = axes[i].imshow(metric_map[:, :, i], cmap=cmap)
            axes[i].set_title(f'{self.get_label(i)}', fontsize=10)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(num_endmembers, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{metric_name} Distance Maps', fontsize=14, y=0.995)
        plt.tight_layout()
        plt.show()
    
    def plot_all_metrics(self, metrics_dict, figsize=(30, 12), cmap='viridis'):
        """
        Plot multiple metrics in a grid.
        
        Args:
            metrics_dict (dict): Dictionary of metric names to plot
            figsize (tuple): Figure size
            cmap (str): Colormap for visualization
        """
        all_maps = {**self.distance_maps, **self.similarity_maps}
        num_metrics = len(metrics_dict)
        metric_names = list(metrics_dict.keys())
        num_endmembers = list(all_maps.values())[0].shape[2]
        
        fig, axes = plt.subplots(num_metrics, num_endmembers, figsize=figsize)
        
        for row, metric_name in enumerate(metric_names):
            metric_map = all_maps[metric_name]
            for col in range(num_endmembers):
                im = axes[row, col].imshow(metric_map[:, :, col], cmap=cmap)
                axes[row, col].set_title(f'{metric_name} - {self.get_label(col)}', fontsize=8)
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], orientation='vertical', fraction=0.02, pad=0.1)
        
        plt.tight_layout()
        plt.show()
    
    # ==================== Save Results ====================
    
    def save_maps(self, output_dir='Outputs', dpi=200):
        """
        Save all computed maps to files.
        
        Args:
            output_dir (str): Directory to save output files
            dpi (int): Resolution for saved images
        """
        os.makedirs(output_dir, exist_ok=True)
        all_maps = {**self.distance_maps, **self.similarity_maps}
        
        for metric_name, metric_map in all_maps.items():
            num_endmembers = metric_map.shape[2]
            ncols = 4
            nrows = (num_endmembers + ncols - 1) // ncols
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12))
            axes = axes.flatten()
            
            for i in range(num_endmembers):
                im = axes[i].imshow(metric_map[:, :, i], cmap='viridis')
                axes[i].set_title(f'{self.get_label(i)}', fontsize=10)
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            for i in range(num_endmembers, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f'{metric_name} Distance Maps', fontsize=14, y=0.995)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{metric_name}_12maps.png', dpi=dpi, bbox_inches='tight')
            plt.close()
        
        print(f"All maps saved to {output_dir}/ directory!")

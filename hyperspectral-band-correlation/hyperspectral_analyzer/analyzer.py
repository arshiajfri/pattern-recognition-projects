# src/analyzer.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import sys

class HyperspectralAnalyzer:
    """
    A class to run a hyperspectral band analysis pipeline, including:
    1. Loading and reshaping data.
    2. Calculating full-band correlation matrices (Pearson, Spearman).
    3. Selecting bands based on a correlation threshold.
    4. Visualizing the correlation matrix of the selected bands.
    """
    
    def __init__(self, config: dict):
        """
        Initializes the analyzer with a configuration dictionary.
        """
        self.config = config
        self.output_dir = self.config['output_dir']
        
        # --- Internal State Attributes ---
        # These will be populated by the pipeline steps
        self.data_cube = None
        self.data_2d = None
        self.num_bands_original = 0
        
        self.pearson_matrix_all = None
        self.spearman_matrix_all = None
        
        self.selected_band_indices = []
        self.data_2d_selected = None
        self.pearson_matrix_selected = None

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        print("HyperspectralAnalyzer initialized.")
        print(f"Output directory set to: {self.output_dir}")

    # --- 1. Data Loading and Preparation ---

    def _load_mat_file(self, file_path: str, data_key: str) -> np.ndarray | None:
        """Private helper to load a .mat file."""
        try:
            mat_data = loadmat(file_path)
            print(f"  Available keys in {file_path}: {list(mat_data.keys())}")
            if data_key not in mat_data:
                print(f"  Error: Key '{data_key}' not found.", file=sys.stderr)
                return None
            return mat_data[data_key]
        except FileNotFoundError:
            print(f"  Error: File not found at {file_path}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"  An error occurred loading {file_path}: {e}", file=sys.stderr)
            return None

    def load_and_prepare_data(self) -> bool:
        """
        Loads the hyperspectral data cube and reshapes it to 2D.
        Returns True on success, False on failure.
        """
        print("\n--- 1. Loading and Preparing Data ---")
        self.data_cube = self._load_mat_file(
            self.config['file_path'], 
            self.config['data_key']
        )
        
        if self.data_cube is None:
            print("Failed to load data cube. Aborting.", file=sys.stderr)
            return False
        
        print(f"Data cube loaded. Shape: {self.data_cube.shape}")
        
        # Reshape data
        h, w, b = self.data_cube.shape
        self.num_bands_original = b
        self.data_2d = self.data_cube.reshape((h * w, b)).astype(np.float64)
        print(f"Data reshaped to (Pixels, Bands): {self.data_2d.shape}")
        return True

    # --- 2. Full-Band Analysis ---

    def _calculate_correlation(self, data: np.ndarray, method: str) -> pd.DataFrame:
        """Private helper to calculate correlation."""
        print(f"  Calculating {method.capitalize()} Correlation Matrix...")
        df = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
        return df.corr(method=method)

    def _plot_heatmap(self, matrix: pd.DataFrame, title: str, filename: str):
        """Private helper to plot and save a heatmap."""
        print(f"  Generating heatmap: {title}")
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, cmap='viridis', xticklabels='auto', yticklabels='auto')
        plt.title(title, fontsize=16)
        plt.xlabel('Band Index', fontsize=12)
        plt.ylabel('Band Index', fontsize=12)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        try:
            plt.savefig(output_path)
            print(f"  Heatmap saved to '{output_path}'")
        except Exception as e:
            print(f"  Error saving heatmap: {e}", file=sys.stderr)
        plt.close()

    def run_full_band_analysis(self):
        """Calculates and plots correlations for all bands."""
        print("\n--- 2. Running Full-Band Analysis ---")
        if self.data_2d is None:
            print("Error: Data not loaded. Run load_and_prepare_data() first.", file=sys.stderr)
            return

        title_suffix = f"(All {self.num_bands_original} Bands)"
        
        # Pearson
        self.pearson_matrix_all = self._calculate_correlation(self.data_2d, 'pearson')
        self._plot_heatmap(
            self.pearson_matrix_all,
            f"Heatmap - Pearson Correlation {title_suffix}",
            "pearson_correlation_all.png"
        )
        
        # Spearman
        self.spearman_matrix_all = self._calculate_correlation(self.data_2d, 'spearman')
        self._plot_heatmap(
            self.spearman_matrix_all,
            f"Heatmap - Spearman Correlation {title_suffix}",
            "spearman_correlation_all.png"
        )

    # --- 3. Band Selection ---

    def select_bands(self):
        """Selects bands by removing highly correlated ones."""
        print("\n--- 3. Selecting Bands ---")
        if self.pearson_matrix_all is None:
            print("Error: Pearson matrix not calculated. Run run_full_band_analysis() first.", file=sys.stderr)
            return
            
        threshold = self.config['correlation_threshold']
        print(f"Using correlation threshold: {threshold}")

        corr_matrix_abs = self.pearson_matrix_all.abs()
        cols = corr_matrix_abs.columns
        to_remove_names = set()

        for i in range(len(cols)):
            if cols[i] in to_remove_names:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in to_remove_names:
                    continue
                if corr_matrix_abs.iloc[i, j] > threshold:
                    to_remove_names.add(cols[j])
        
        all_band_names = set(self.pearson_matrix_all.columns)
        selected_band_names = sorted(list(all_band_names - to_remove_names))
        self.selected_band_indices = [int(name) for name in selected_band_names]
        
        self._print_band_selection_report(len(to_remove_names))

    def _print_band_selection_report(self, num_removed: int):
        """Prints a summary of the band selection results."""
        print("\n--- Band Selection Report ---")
        print(f"Correlation Threshold: {self.config['correlation_threshold']}")
        print(f"Total Original Bands:  {self.num_bands_original}")
        print(f"Total Bands Removed:   {num_removed}")
        print(f"Total Bands Selected:  {len(self.selected_band_indices)}")
        print("---------------------------------")
        print(f"List of {len(self.selected_band_indices)} SELECTED band indices:")
        print(self.selected_band_indices)
        print("---------------------------------")

    # --- 4. Post-Selection Analysis ---

    def run_selected_band_analysis(self):
        """Calculates and plots correlation for *selected* bands."""
        print("\n--- 4. Visualizing Post-Selection Correlation ---")
        if not self.selected_band_indices:
            print("Error: Bands not selected. Run select_bands() first.", file=sys.stderr)
            return

        self.data_2d_selected = self.data_2d[:, self.selected_band_indices]
        print(f"New data shape (Pixels, Selected Bands): {self.data_2d_selected.shape}")

        self.pearson_matrix_selected = self._calculate_correlation(self.data_2d_selected, 'pearson')
        
        num_selected = len(self.selected_band_indices)
        threshold = self.config['correlation_threshold']
        
        self._plot_heatmap(
            self.pearson_matrix_selected,
            f"Heatmap - Pearson (After Selection, {num_selected} Bands, Thresh={threshold})",
            "pearson_correlation_selected.png"
        )

    # --- 5. Main Pipeline Execution ---

    def run_pipeline(self):
        """
        Runs the full analysis pipeline from start to finish.
        """
        print("===== Starting Hyperspectral Analysis Pipeline =====")
        
        if not self.load_and_prepare_data():
            print("===== Pipeline failed at data loading. =====", file=sys.stderr)
            return
            
        self.run_full_band_analysis()
        self.select_bands()
        self.run_selected_band_analysis()
        
        print("\n===== Pipeline complete. Results saved in 'results' directory. =====")
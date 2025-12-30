# src/analyzer.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

class HyperspectralAnalyzer:
    """
    A class to run a hyperspectral analysis pipeline, including:
    1. Loading data and ground truth.
    2. Calculating correlation matrices (Pearson, Spearman).
    3. Selecting bands based on a correlation threshold for each metric.
    4. Running KNN classification with the selected bands.
    5. Reporting accuracy and visualizing the final classification map.
    """
    
    def __init__(self, config: dict):
        """
        Initializes the analyzer with a configuration dictionary.
        """
        self.config = config
        self.output_dir = self.config['output_dir']
        
        # --- Internal State Attributes ---
        self.data_cube = None
        self.gt_cube = None
        self.data_2d = None
        self.gt_1d = None
        self.height = 0
        self.width = 0
        self.num_bands_original = 0
        
        self.pearson_matrix_all = None
        self.spearman_matrix_all = None

        os.makedirs(self.output_dir, exist_ok=True)
        print("HyperspectralAnalyzer initialized.")
        print(f"Output directory set to: {self.output_dir}")

    # --- 1. Data Loading and Preparation ---

    def _load_mat_file(self, file_path: str, data_key: str) -> np.ndarray | None:
        """Private helper to load a .mat file."""
        try:
            mat_data = loadmat(file_path)
            if data_key not in mat_data:
                print(f"  Error: Key '{data_key}' not found in {file_path}.", file=sys.stderr)
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
        Loads the hyperspectral data and ground truth, then reshapes them.
        Returns True on success, False on failure.
        """
        print("\n--- 1. Loading and Preparing Data ---")
        # Load hyperspectral data
        self.data_cube = self._load_mat_file(self.config['file_path'], self.config['data_key'])
        if self.data_cube is None:
            print("Failed to load data cube. Aborting.", file=sys.stderr)
            return False
        print(f"Data cube loaded. Shape: {self.data_cube.shape}")

        # Load ground truth data
        self.gt_cube = self._load_mat_file(self.config['gt_file_path'], self.config['gt_data_key'])
        if self.gt_cube is None:
            print("Failed to load ground truth. Aborting.", file=sys.stderr)
            return False
        print(f"Ground truth loaded. Shape: {self.gt_cube.shape}")
        
        # Reshape data
        self.height, self.width, self.num_bands_original = self.data_cube.shape
        self.data_2d = self.data_cube.reshape((self.height * self.width, self.num_bands_original)).astype(np.float64)
        self.gt_1d = self.gt_cube.reshape((self.height * self.width,)).astype(int)
        
        print(f"Data reshaped to (Pixels, Bands): {self.data_2d.shape}")
        print(f"Ground truth reshaped to (Pixels,): {self.gt_1d.shape}")
        return True

    # --- 2. Correlation Analysis ---

    def _calculate_correlation(self, data: np.ndarray, method: str) -> pd.DataFrame:
        """Private helper to calculate correlation."""
        print(f"  Calculating {method.capitalize()} Correlation Matrix...")
        df = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
        return df.corr(method=method)

    def _plot_heatmap(self, matrix: pd.DataFrame, title: str, filename: str):
        """Private helper to plot and save a heatmap."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, cmap='viridis')
        plt.title(title, fontsize=16)
        plt.xlabel('Band Index', fontsize=12)
        plt.ylabel('Band Index', fontsize=12)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        print(f"  Heatmap saved to '{output_path}'")
        plt.close()

    def run_correlation_analysis(self):
        """Calculates correlations for all bands using both metrics."""
        print("\n--- 2. Running Full-Band Correlation Analysis ---")
        if self.data_2d is None:
            print("Error: Data not loaded.", file=sys.stderr)
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

    # --- 3. Feature (Band) Selection ---

    def select_bands_from_matrix(self, corr_matrix: pd.DataFrame, metric_name: str) -> list[int]:
        """Selects bands by removing highly correlated ones based on the given matrix."""
        print(f"\n--- 3. Selecting Bands based on {metric_name.capitalize()} Correlation ---")
        threshold = self.config['correlation_threshold']
        print(f"Using correlation threshold: {threshold}")

        corr_matrix_abs = corr_matrix.abs()
        cols = corr_matrix_abs.columns
        to_remove = set()

        for i in range(len(cols)):
            if cols[i] in to_remove:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in to_remove:
                    continue
                if corr_matrix_abs.iloc[i, j] > threshold:
                    to_remove.add(cols[j])
        
        selected_band_names = sorted(list(set(corr_matrix.columns) - to_remove))
        selected_band_indices = [int(name) for name in selected_band_names]
        
        print(f"Total Original Bands: {self.num_bands_original}")
        print(f"Total Bands Removed:  {len(to_remove)}")
        print(f"Total Bands Selected: {len(selected_band_indices)}")
        print(f"Selected band indices: {selected_band_indices}")
        
        return selected_band_indices

    # --- 4. KNN Classification ---

    def run_classification(self, selected_bands: list[int], metric_name: str):
        """
        Runs KNN classification using the provided list of selected bands.
        """
        print(f"\n--- 4. Running KNN Classification for {metric_name.capitalize()} Selected Bands ---")
        
        # 1. Prepare data and labels
        # Filter out unlabeled pixels (where ground truth is 0)
        labeled_indices = np.where(self.gt_1d > 0)[0]
        X_labeled = self.data_2d[labeled_indices]
        y_labeled = self.gt_1d[labeled_indices]

        # Select the feature-selected bands
        X_selected = X_labeled[:, selected_bands]
        
        # 2. Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        # 3. Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, 
            y_labeled, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state'],
            stratify=y_labeled # Ensures proportional class representation
        )
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Test set size:     {X_test.shape[0]} samples")

        # 4. Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=self.config['knn_neighbors'])
        print(f"Training KNN with k={self.config['knn_neighbors']}...")
        knn.fit(X_train, y_train)

        # 5. Evaluate the model
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n--- Classification Report (Metric: {metric_name}) ---")
        print(f"Accuracy on Test Set: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        print("----------------------------------------------------")

        # 6. Generate and save the final classification map
        # Scale all data (including unlabeled) before prediction
        full_data_selected = self.data_2d[:, selected_bands]
        full_data_scaled = scaler.transform(full_data_selected) # Use the same scaler
        
        print("Generating full classification map...")
        full_prediction = knn.predict(full_data_scaled)
        
        # Create the map, ensuring unlabeled pixels remain 0
        classification_map = full_prediction.reshape(self.height, self.width)
        classification_map[self.gt_cube == 0] = 0
        
        # Plot and save the map
        plt.figure(figsize=(10, 8))
        plt.imshow(classification_map, cmap='viridis')
        plt.title(f'KNN Classification Map ({metric_name}, Acc: {accuracy:.3f})', fontsize=16)
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.colorbar(ticks=range(np.max(self.gt_cube) + 1))
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f"classification_map_{metric_name.lower()}.png")
        plt.savefig(output_path)
        print(f"Classification map saved to '{output_path}'")
        plt.close()

    # --- 5. Main Pipeline Execution ---

    def run_pipeline(self):
        """
        Runs the full analysis pipeline from start to finish.
        """
        print("===== Starting Hyperspectral Analysis & Classification Pipeline =====")
        
        if not self.load_and_prepare_data():
            print("===== Pipeline failed at data loading. ====", file=sys.stderr)
            return
            
        self.run_correlation_analysis()
        
        # --- Run pipeline for Pearson metric ---
        pearson_bands = self.select_bands_from_matrix(self.pearson_matrix_all, "pearson")
        self.run_classification(pearson_bands, "pearson")
        
        # --- Run pipeline for Spearman metric ---
        spearman_bands = self.select_bands_from_matrix(self.spearman_matrix_all, "spearman")
        self.run_classification(spearman_bands, "spearman")

        # --- Run pipeline for All Bands (no feature selection) ---
        print("\n--- Running Classification on ALL BANDS for comparison ---")
        all_bands = list(range(self.num_bands_original))
        print(f"Total Bands Selected: {len(all_bands)}")
        self.run_classification(all_bands, "all_bands")
        
        print("\n===== Pipeline complete. Results saved in 'results' directory. =====")

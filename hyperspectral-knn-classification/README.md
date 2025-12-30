# Hyperspectral KNN Classification with Dimensionality Reduction

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A complete pipeline for **hyperspectral image classification** using K-Nearest Neighbors (KNN) with correlation-based dimensionality reduction. This project compares classification accuracy using all bands versus reduced band sets selected via Pearson and Spearman correlation analysis.

## Features

- **End-to-End Pipeline**: From data loading to classification and visualization.
- **Dimensionality Reduction**: Implements feature selection by removing highly correlated bands using a configurable threshold.
- **Correlation Analysis**: Utilizes both Pearson and Spearman correlation metrics to analyze band relationships.
- **KNN Classification**: Employs the KNN algorithm for pixel-wise classification.
- **Comparative Analysis**: Compares the classification results of using all bands versus feature-selected bands.
- **Visualization**: Generates and saves classification maps and correlation heatmaps.
- **Modular Code**: Organized into a clear and reusable `HyperspectralAnalyzer` class.

## Pipeline

The project follows these steps:

1.  **Load Data**: The Indian Pines hyperspectral data cube and its corresponding ground truth map are loaded from `.mat` files.
2.  **Data Preparation**: The 3D data cube (Height x Width x Bands) is reshaped into a 2D matrix (Pixels x Bands) for easier processing.
3.  **Correlation Analysis**:
    *   Pearson and Spearman correlation matrices are calculated for all spectral bands.
    *   Heatmaps of these matrices are generated and saved to the `results/` directory.
4.  **Band Selection (Dimensionality Reduction)**:
    *   A subset of bands is selected by removing those with a correlation higher than a specified threshold (e.g., 0.95).
    *   This process is done independently for both Pearson and Spearman correlations.
5.  **KNN Classification**:
    *   The KNN classifier is trained and evaluated on three different sets of features:
        1.  **All Bands**: Using the original, full set of spectral bands.
        2.  **Pearson-Selected Bands**: Using the reduced set of bands from the Pearson correlation analysis.
        3.  **Spearman-Selected Bands**: Using the reduced set of bands from the Spearman correlation analysis.
    *   For each case, the data is split into training and testing sets, and the model's accuracy is evaluated.
6.  **Generate Classification Maps**:
    *   A full classification map of the entire scene is generated for each of the three feature sets.
    *   These maps are saved as PNG images in the `results/` directory.

## Project Structure

```
hyperspectral-knn-classification/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
├── knn_classifier/                     # HyperspectralAnalyzer module directory
│   ├── __init__.py                    # Module init file
│   ├── config.py                      # Configuration settings
│   ├── analyzer.py                    # Main HyperspectralAnalyzer class
│   └── main.py                        # Main executable script
├── Data/                               # Data directory
│   ├── Indian_pines_corrected.mat     # Hyperspectral data cube
│   └── Indian_pines_gt.mat            # Ground truth labels
└── results/                            # Output plots and classification maps
```

## Requirements

### Python Packages

```bash
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

### Installation

```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn
```

## Usage

### Basic Usage

```python
from knn_classifier import HyperspectralAnalyzer, CONFIG

# Create an instance
analyzer = HyperspectralAnalyzer(CONFIG)

# Run the pipeline
analyzer.run_pipeline()

# Access the results
print(f"Original number of bands: {analyzer.num_bands_original}")
```

### Configure the Analysis

Modify the settings in `knn_classifier/config.py`:

```python
CONFIG = {
    "file_path": "Data/Indian_pines_corrected.mat",
    "gt_file_path": "Data/Indian_pines_gt.mat",
    "output_dir": "results",
    "correlation_threshold": 0.95,
    "knn_neighbors": 5,
    "test_size": 0.2,
    "random_state": 42
}
```

### Run from Command Line

```bash
python knn_classifier/main.py
```

## Results

The script will generate the following files in the `results/` directory:

-   `pearson_correlation_all.png`: A heatmap of the Pearson correlation matrix for all bands.
-   `spearman_correlation_all.png`: A heatmap of the Spearman correlation matrix for all bands.
-   `classification_map_all_bands.png`: The final classification map using all spectral bands.
-   `classification_map_pearson.png`: The final classification map using the bands selected after Pearson correlation analysis.
-   `classification_map_spearman.png`: The final classification map using the bands selected after Spearman correlation analysis.

The console output will display the classification accuracy and a detailed report for each of the three classification runs.

## Dataset

This project uses the **Indian Pines** dataset, a well-known benchmark for hyperspectral image analysis.

-   **Source**: Captured by the Airborne Visible/Infrared Imaging Spectrometer (AVIRIS) sensor over the Indian Pines test site in Northwestern Indiana.
-   **Data**: `Indian_pines_corrected.mat` contains the hyperspectral data cube with dimensions 145x145 pixels and 220 spectral bands.
-   **Ground Truth**: `Indian_pines_gt.mat` provides the ground truth with 16 different land cover classes.

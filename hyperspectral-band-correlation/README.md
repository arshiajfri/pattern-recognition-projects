# Hyperspectral Band Correlation Analysis

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An object-oriented Python framework for analyzing **hyperspectral band-to-band correlation**. The primary goal is to identify and remove redundant spectral bands by applying a correlation threshold, effectively performing dimensionality reduction while preserving essential spectral information.

## What is Band Correlation Analysis?

In hyperspectral imaging, many adjacent bands are highly correlated (e.g., \>95% similar). This is known as **multicollinearity** or **data redundancy**.

This redundancy poses several problems:

  - **Curse of Dimensionality:** Too many features (bands) for too few training samples.
  - **Model Instability:** Many classification algorithms (like SVMs) perform poorly with highly correlated features.
  - **Computational Cost:** Processing hundreds of bands is slow.

Band correlation analysis identifies and removes these redundant bands, creating a smaller, more efficient, and more robust dataset for analysis.

## Project Structure

```
hyperspectral-band-correlation/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
├── hyperspectral_analyzer/             # HyperspectralAnalyzer module directory
│   ├── __init__.py                    # Module init file
│   ├── config.py                      # Configuration settings
│   ├── analyzer.py                    # Main HyperspectralAnalyzer class
│   └── main.py                        # Main executable script
├── Data/                               # Data directory
│   └── Indian_pines_corrected.mat     # Hyperspectral data cube
└── results/                            # Output plots are saved here
```

## Requirements

### Python Packages

```bash
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

### Installation

```bash
pip install numpy scipy pandas matplotlib seaborn
```

## Features

### HyperspectralAnalyzer Class

A reusable Python class that encapsulates the entire analysis pipeline.

**Key Methods:**

  - `load_and_prepare_data()`: Loads the `.mat` data cube and reshapes it for analysis.
  - `run_full_band_analysis()`: Computes and plots Pearson and Spearman correlation matrices for all bands.
  - `select_bands()`: Selects a subset of bands based on the correlation threshold.
  - `run_selected_band_analysis()`: Computes and plots a new correlation matrix for *only* the selected bands to validate the process.
  - `plot_spectral_signatures()`: (Optional) Plots spectral signatures to visually confirm information preservation.
  - `run_pipeline()`: Executes the entire workflow from start to finish.

**Attributes:**

  - `data_cube`: The original 3D (H, W, Bands) data.
  - `data_2d`: The reshaped 2D (Pixels, Bands) data.
  - `pearson_matrix_all`: The correlation matrix for all 220 bands.
  - `selected_band_indices`: A list of the final selected band indices.
  - `pearson_matrix_selected`: The new correlation matrix for the selected subset.

## Usage

### Basic Usage

```python
from hyperspectral_analyzer import HyperspectralAnalyzer, CONFIG

# Create an instance
analyzer = HyperspectralAnalyzer(CONFIG)

# Run the pipeline
analyzer.run_pipeline()

# Access the results
print(f"Original number of bands: {analyzer.num_bands_original}")
print(f"Selected number of bands: {len(analyzer.selected_band_indices)}")

# Get the final, reduced 2D data
reduced_data = analyzer.data_2d_selected
print(f"Shape of reduced data: {reduced_data.shape}")
```

### Configure the Analysis

Modify the settings in `hyperspectral_analyzer/config.py` to match your data and parameters.

```python
CONFIG = {
    "file_path": "Data/Indian_pines_corrected.mat",
    "data_key": "indian_pines_corrected",
    "output_dir": "results",
    "correlation_threshold": 0.95  # 0.95 is aggressive, 0.99 is conservative
}
```

### Run from Command Line

```bash
python hyperspectral_analyzer/main.py
```

## Methodology

### 1\. Correlation Computation

1.  **Load Data**: The 3D (145x145x220) data cube is loaded.
2.  **Reshape**: Data is reshaped into a 2D (21025, 220) matrix, where each row is a pixel and each column is a band.
3.  **Matrix Calculation**: `pandas.DataFrame.corr()` is used to compute two 220x220 matrices:
      * **Pearson Correlation**: Measures *linear* relationships.
      * **Spearman Correlation**: Measures *monotonic* relationships (rank-based).

### 2\. Band Selection

1.  **Thresholding**: The script iterates through the absolute Pearson correlation matrix.
2.  **Iteration**: It compares every unique pair of bands (i, j).
3.  **Removal**: If `correlation(i, j) > threshold` (e.g., 0.95), one of the bands (band `j`) is added to a `to_remove` set.
4.  **Selection**: The final list of selected bands is the set of all bands *minus* the `to_remove` set.

## Visualization

The pipeline generates three key visualizations saved in the `results/` folder:

1.  **Full Correlation Heatmap**: A heatmap of the original 220x220 Pearson matrix. Bright yellow squares clearly show large blocks of highly correlated bands.
2.  **Selected Bands Heatmap**: A new heatmap computed using *only* the selected bands (e.g., 85x85). This plot should appear much darker, proving that the high-correlation blocks have been successfully removed.
3.  **Spectral Signature Plot**: Compares the full spectrum (all 220 bands) for a specific land cover (e.g., "Corn") against the points from the selected bands. This visually confirms that the selected subset still captures the unique spectral "fingerprint" of the material.

## Dataset

This project is configured to use the well-known **Indian Pines** hyperspectral dataset.

  - **Source**: Airborne Visible/Infrared Imaging Spectrometer (AVIRIS) sensor.
  - **Data**: `Indian_pines_corrected.mat` (145x145 pixels, 220 spectral bands).
  - **Ground Truth**: `Indian_pines_gt.mat` (16 classes of agricultural/natural land cover).
  - **Context**: It is a standard benchmark dataset for hyperspectral image classification and feature selection research.

## Applications

This code serves as a critical **preprocessing step** for any hyperspectral analysis workflow:

  - **Dimensionality Reduction**: Directly reduces the number of features.
  - **Classification Improvement**: Improves the accuracy and stability of ML models (SVM, k-NN, Random Forest) by removing multicollinearity.
  - **Feature Selection**: Provides a fast, filter-based method for feature selection.
  - **Data Visualization**: Simplifies the process of understanding data redundancy.

## License

MIT License - See [LICENSE](LICENSE) file.
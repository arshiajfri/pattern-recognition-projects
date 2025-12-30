# Hyperspectral Image Classification using Distance Metrics

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Python implementation for **hyperspectral image classification** using various distance and similarity metrics. This project compares multiple spectral matching techniques for mineral identification in the Cuprite hyperspectral dataset.

## What is Spectral Matching?

Spectral matching is a technique used in hyperspectral image analysis to identify materials by comparing pixel spectra against known reference signatures (endmembers). Different distance/similarity metrics capture different aspects of spectral similarity.

## Project Structure

```
hyperspectral-distance-metrics/
├── README.md                           # This file
├── main.ipynb                          # Main notebook with sample usage
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
├── hyperspectral_classifier/           # HyperspectralClassifier module directory
│   ├── __init__.py                    # Module init file
│   └── classifier.py                  # Main HyperspectralClassifier class
├── Data/                               # Data directory
│   ├── Cuprite_S1_F224/               # ENVI format hyperspectral image
│   └── groundTruth_Cuprite_nEnd12.mat # Reference spectral signatures
└── Outputs/                            # Generated distance maps
```

## Requirements

### Python Packages

```bash
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
spectral>=0.23.0
jupyter>=1.0.0
```

### Installation

```bash
pip install numpy scipy matplotlib spectral jupyter
```

## Features

### HyperspectralClassifier Class

A reusable Python class that performs hyperspectral image classification using multiple distance metrics.

**Key Methods:**
- `compute_all()`: Computes all distance and similarity metrics
- `compute_euclidean()`, `compute_cosine()`, `compute_sam()`, etc.: Individual metric computation
- `plot_metric_maps(metric_name)`: Visualize maps for a specific metric
- `plot_all_metrics(metrics_dict)`: Plot multiple metrics in a grid
- `save_maps(output_dir)`: Save all computed maps to files

**Attributes:**
- `image`: Loaded hyperspectral image
- `spectral_sigs`: Reference spectral signatures
- `labels`: Endmember labels
- `distance_maps`: Dictionary of computed distance maps
- `similarity_maps`: Dictionary of computed similarity maps

### Distance Metrics Implemented

| Metric | Type | Description |
|--------|------|-------------|
| **Euclidean** | Distance | L2 norm - straight-line distance in spectral space |
| **Cosine** | Distance | Angle-based distance (1 - cosine similarity) |
| **Cityblock** | Distance | L1 norm (Manhattan distance) |
| **Canberra** | Distance | Weighted version of Manhattan distance |

### Similarity/Error Metrics Implemented

| Metric | Type | Description |
|--------|------|-------------|
| **SAM** | Similarity | Spectral Angle Mapper - angle between spectral vectors |
| **SCC** | Similarity | Spectral Correlation Coefficient - Pearson correlation |
| **MSE** | Error | Mean Squared Error between spectra |
| **RMSE** | Error | Root Mean Squared Error |

## Usage

### Basic Usage

```python
from hyperspectral_classifier import HyperspectralClassifier

# Create classifier instance
classifier = HyperspectralClassifier(
    image_path="Data/Cuprite_S1_F224/Cuprite_S1_F224.img",
    header_path="Data/Cuprite_S1_F224/Cuprite_S1_F224.hdr",
    signatures_path="Data/groundTruth_Cuprite_nEnd12.mat"
)

# Compute all metrics
classifier.compute_all()

# Plot results
classifier.plot_metric_maps('SAM')
```

### Working with Individual Metrics

```python
# Create classifier instance
classifier = HyperspectralClassifier(image_path, header_path, signatures_path)

# Compute specific metrics
classifier.compute_euclidean()
classifier.compute_sam()
classifier.compute_scc()

# Access results
euclidean_map = classifier.distance_maps['Euclidean']
sam_map = classifier.similarity_maps['SAM']

# Save all maps to files
classifier.save_maps(output_dir='Outputs')
```

## Methodology

1. **Data Loading**: Load ENVI format hyperspectral image and reference signatures
2. **Normalization**: Min-max normalize the image data
3. **Distance Computation**: Compute pixel-wise distances to all reference signatures
4. **Visualization**: Generate distance/similarity maps for each endmember
5. **Classification**: Assign pixels to the closest endmember (minimum distance)

## Dataset

The project uses the **Cuprite** hyperspectral dataset:
- **Image size**: 250 × 190 pixels (47,500 total)
- **Spectral bands**: 224 bands
- **Endmembers**: 12 mineral reference signatures

Minerals included:
- Alunite, Andradite, Buddingtonite, Dumortierite
- Kaolinite (2 variants), Muscovite, Montmorillonite
- Nontronite, Pyrope, Sphene, Chalcedony

## Output

The notebook generates distance/similarity maps for each metric and endmember combination, saved as PNG files in the `Outputs/` directory.

## Applications

- Mineral mapping and identification
- Remote sensing image classification
- Material detection in hyperspectral imagery
- Comparative analysis of spectral matching algorithms

## References

- Spectral Angle Mapper (SAM) - Kruse et al., 1993
- Spectral Information Divergence - Chang, 2000
- Various distance metrics for hyperspectral analysis

## License

MIT License - See [LICENSE](LICENSE) file.

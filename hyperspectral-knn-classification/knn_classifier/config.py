# knn_classifier/config.py

"""
Configuration settings for the hyperspectral analysis pipeline.
"""

import os

# Get the directory where this config file is located
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

CONFIG = {
    # --- Paths ---
    "file_path": os.path.join(_PROJECT_ROOT, "Data", "Indian_pines_corrected.mat"),
    "data_key": "indian_pines_corrected",
    "gt_file_path": os.path.join(_PROJECT_ROOT, "Data", "Indian_pines_gt.mat"),
    "gt_data_key": "indian_pines_gt",
    "output_dir": os.path.join(_PROJECT_ROOT, "results"),
    
    # --- Analysis Parameters ---
    "correlation_threshold": 0.95,

    # --- Classification Parameters ---
    "knn_neighbors": 5,
    "test_size": 0.2,
    "random_state": 42  # for reproducibility
}
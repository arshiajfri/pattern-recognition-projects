# hyperspectral_analyzer/config.py

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
    "output_dir": os.path.join(_PROJECT_ROOT, "results"),
    
    # --- Analysis Parameters ---
    "correlation_threshold": 0.95
}
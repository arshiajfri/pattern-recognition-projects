# src/config.py

"""
Configuration settings for the hyperspectral analysis pipeline.
"""

CONFIG = {
    # --- Paths ---
    "file_path": "Data/Indian_pines_corrected.mat",
    "data_key": "indian_pines_corrected",
    "output_dir": "results",
    
    # --- Analysis Parameters ---
    "correlation_threshold": 0.95
}
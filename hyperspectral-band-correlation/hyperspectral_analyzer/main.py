# src/main.py

from analyzer import HyperspectralAnalyzer
from config import CONFIG
import sys

def main():
    """
    Main execution function.
    Initializes the analyzer with the global config and runs the pipeline.
    """
    print("Running main execution...")
    try:
        # 1. Create an instance of the analyzer using the config
        analyzer = HyperspectralAnalyzer(CONFIG)
        
        # 2. Run the entire pipeline
        analyzer.run_pipeline()
        
    except KeyError as e:
        print(f"FATAL ERROR: Missing key in CONFIG: {e}", file=sys.stderr)
        print("Please check your 'src/config.py' file.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
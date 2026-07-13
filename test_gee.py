import sys
import os
from pathlib import Path

# Add src directory to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

import ee
from config_manager import load_config

def main():
    try:
        print("Loading configuration...")
        config = load_config()
        project_id = config.get("gee_project_id")
        
        if not project_id:
            print("Error: gee_project_id is not set in config/settings.json")
            sys.exit(1)
            
        print(f"Initializing Earth Engine with project ID: {project_id}")
        ee.Initialize(project=project_id)
        
        print("Google Earth Engine connected successfully")
        
        print("Testing satellite data access...")
        dataset = ee.ImageCollection("MODIS/061/MOD13Q1")
        info = dataset.first().getInfo()
        
        if info:
            print("Satellite data access working")
        else:
            print("Error: Could not retrieve data info")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

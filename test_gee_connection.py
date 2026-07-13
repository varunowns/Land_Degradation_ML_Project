import ee
from src.config_manager import load_config

config = load_config()

project_id = config.get("gee_project_id", "")

print("Using project:", project_id)

try:
    if not project_id:
        print("Error: Google Cloud Project ID missing")
    else:
        print("Attempting authentication...")
        ee.Authenticate(auth_mode='localhost')
        
        print("Initializing...")
        ee.Initialize(project=project_id)
        
        img = (
           ee.ImageCollection("MODIS/061/MOD13Q1")
           .first()
           .getInfo()
        )
        
        print("Google Earth Engine connected successfully")
        print("Satellite data access working")

except Exception as e:
    print(e)

import os
import sys
import urllib.request
from pothole_config import POTHOLE_MODEL_DIR, POTHOLE_MODEL_PATH

def download_pothole_weights():
    os.makedirs(POTHOLE_MODEL_DIR, exist_ok=True)
    url = "https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt"
    
    print(f"Downloading Pothole weights: {url} -> {POTHOLE_MODEL_PATH}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            with open(POTHOLE_MODEL_PATH, 'wb') as f:
                f.write(response.read())
        print("Pothole model weights downloaded successfully!")
    except Exception as e:
        print(f"Failed to download pothole weights: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_pothole_weights()

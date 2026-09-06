import os
import urllib.request
import sys

def download_file(url, dest):
    print(f"Downloading {url} -> {dest}...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        # Use a user-agent to avoid HTTP 403 Forbidden issues on certain servers
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            with open(dest, 'wb') as f:
                f.write(response.read())
        print("Success!")
    except Exception as e:
        print(f"Failed to download: {e}")
        sys.exit(1)

def main():
    # 1. Download test image (standard YOLO test image with a bus and people)
    img_url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
    img_dest = os.path.join("input", "test.jpg")
    download_file(img_url, img_dest)
    
    # 2. Download road video (Traffic.mp4 from DeGirum PySDKExamples)
    video_url = "https://github.com/DeGirum/PySDKExamples/raw/main/images/Traffic.mp4"
    video_dest = os.path.join("input", "road_video.mp4")
    download_file(video_url, video_dest)
    
    # 3. Ensure models/ directory exists and download YOLO11 nano weights
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    model_dest = os.path.join(models_dir, "yolo11n.pt")
    download_file(model_url, model_dest)
    
    print("Asset download completed. Ready for Phase 1 testing!")

if __name__ == "__main__":
    main()

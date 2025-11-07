import os
import shutil
from pathlib import Path

def download_datasets(data_folder):
    # Check if we already have video files
    data_path = Path(data_folder)
    if data_path.exists():
        mp4_files = list(data_path.rglob("*.mp4"))
        if len(mp4_files) > 0:
            print(f"Found {len(mp4_files)} videos already downloaded, skipping...")
            return
    
    # Try to import kagglehub
    try:
        import kagglehub
    except:
        print("kagglehub not installed. Run: pip install kagglehub")
        return
    
    # List of datasets to download
    datasets = [
        ("nikospetrellis/nitymed", "nitymed"),
        ("ismailnasri20/driver-drowsiness-dataset-ddd", "ddd"),
        ("esrakavalci/sust-ddd", "sust_ddd")
    ]
    
    # Download each dataset
    for dataset_name, folder_name in datasets:
        print(f"Downloading {dataset_name}...")
        
        try:
            path = kagglehub.dataset_download(dataset_name)
            print(f"Downloaded to: {path}")
            
            # Copy files to our data folder
            dest_folder = data_path / folder_name
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            src = Path(path)
            for file in src.rglob("*"):
                if file.is_file():
                    relative = file.relative_to(src)
                    destination = dest_folder / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        shutil.copy2(file, destination)
            
            print(f"Copied to {dest_folder}")
            
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
    
    print("Done downloading datasets!")

import os
import shutil
from pathlib import Path

def collect_v04_generations():
    # Define paths
    project_root = Path(__file__).resolve().parents[2]
    source_dir = project_root / "synthetic_data" / "tiles" / "v04"
    target_dir = project_root / "synthetic_data" / "datasets" / "v04" / "generations"

    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist.")
        return

    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {target_dir}")

    # Iterate through subdirectories
    count = 0
    for item in sorted(source_dir.iterdir()):
        if item.is_dir():
            source_file = item / "generation.png"
            if source_file.exists():
                # New filename based on subdirectory name
                new_filename = f"{item.name}.png"
                target_path = target_dir / new_filename
                
                # Copy file
                shutil.copy2(source_file, target_path)
                print(f"Copied: {item.name}/generation.png -> {new_filename}")
                count += 1
            else:
                # print(f"Skipping {item.name}: generation.png not found")
                pass
    
    print(f"\nSuccess! Collected {count} images to {target_dir}")

if __name__ == "__main__":
    collect_v04_generations()

import os
import shutil
from pathlib import Path

def collect_renders():
    # Define paths
    # Assuming script is in src/isometric_nyc/
    project_root = Path(__file__).resolve().parents[2] 
    source_dir = project_root / "synthetic_data" / "tiles" / "v02"
    # Using 'datasets' instead of 'datsets' (typo correction)
    target_dir = project_root / "synthetic_data" / "datasets" / "v02" / "renders"

    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist.")
        return

    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {target_dir}")

    # Iterate through subdirectories
    count = 0
    # Sort to ensure deterministic order/logging
    for item in sorted(source_dir.iterdir()):
        if item.is_dir():
            render_path = item / "render.png"
            if render_path.exists():
                # New filename based on subdirectory name
                new_filename = f"{item.name}.png"
                target_path = target_dir / new_filename
                
                # Copy file
                shutil.copy2(render_path, target_path)
                print(f"Copied: {item.name}/render.png -> {new_filename}")
                count += 1
            else:
                # Optional: warn if missing
                # print(f"Skipping {item.name}: render.png not found")
                pass
    
    print(f"\nSuccess! Collected {count} images to {target_dir}")

if __name__ == "__main__":
    collect_renders()

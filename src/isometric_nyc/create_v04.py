import os
import shutil
from pathlib import Path

def create_v04_dataset():
    # Define paths
    project_root = Path(__file__).resolve().parents[2]
    tiles_dir = project_root / "synthetic_data" / "tiles"
    v02_dir = tiles_dir / "v02"
    v03_dir = tiles_dir / "v03"
    target_dir = tiles_dir / "v04"

    # Files to copy
    allowed_files = {"generation.png", "render.png", "whitebox.png", "view.json"}

    if target_dir.exists():
        print(f"Warning: Target directory {target_dir} already exists.")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {target_dir}")

    current_index = 0

    # Function to process a source directory
    def process_source_dir(source_dir):
        nonlocal current_index
        if not source_dir.exists():
            print(f"Skipping {source_dir} (does not exist)")
            return

        # Iterate through sorted subdirectories
        # We filter for directories that look like indices (digits) just to be safe, 
        # or just take all subdirectories. The prompt implies taking "all subdirectories".
        # Based on `ls` output, they are named "000", "001", etc.
        items = sorted([item for item in source_dir.iterdir() if item.is_dir()])
        
        for item in items:
            # Check if this directory contains any of the interesting files?
            # Or just copy if files exist.
            
            # Destination path
            dest_subdir_name = f"{current_index:03d}"
            dest_subdir = target_dir / dest_subdir_name
            
            files_copied = 0
            
            for filename in allowed_files:
                src_file = item / filename
                if src_file.exists():
                    if not dest_subdir.exists():
                        dest_subdir.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(src_file, dest_subdir / filename)
                    files_copied += 1
            
            if files_copied > 0:
                print(f"Copied {item.name} from {source_dir.name} to {dest_subdir_name} ({files_copied} files)")
                current_index += 1
            else:
                # print(f"Skipping {item.name} from {source_dir.name} (no relevant files found)")
                pass

    # Process v02 then v03
    print("Processing v02...")
    process_source_dir(v02_dir)
    
    print("Processing v03...")
    process_source_dir(v03_dir)

    print(f"\nSuccess! Created v04 with {current_index} entries.")

if __name__ == "__main__":
    create_v04_dataset()

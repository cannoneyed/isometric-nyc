import csv
import os
from pathlib import Path

def create_v04_infills_csv():
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "synthetic_data" / "datasets" / "v04"
    
    infills_dir = dataset_dir / "infills"
    generations_dir = dataset_dir / "generations"
    output_csv = dataset_dir / "infills_v04.csv"

    prompt_text = "Convert the red outlined quadrants of the image to <isometric nyc pixel art> in precisely the style of the remaining pixel-art quadrants."

    if not infills_dir.exists():
        print(f"Error: Infills directory not found at {infills_dir}")
        return
    if not generations_dir.exists():
        print(f"Error: Generations directory not found at {generations_dir}")
        return

    infill_files = sorted([f.name for f in infills_dir.glob("*.png")])
    print(f"Found {len(infill_files)} infill files.")

    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['template', 'generation', 'prompt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        rows_written = 0
        for filename in infill_files:
            # Extract number prefix (assuming filenames are like 000_01.png or similar, or just 000.png)
            # The prompt says "prefix number of the filename, e.g. 000". 
            # We'll assume the filename might be just "000.png" or "000_something.png".
            # Given previous steps, they are likely "000.png".
            
            # Extract prefix:
            # 000.png -> 000
            # 000_01.png -> 000
            prefix = filename.split('_')[0].split('.')[0]
            
            generation_filename = f"{prefix}.png"
            generation_path = generations_dir / generation_filename
            
            if generation_path.exists():
                writer.writerow({
                    'template': f"infills_v04/{filename}",
                    'generation': f"generations/{generation_filename}",
                    'prompt': prompt_text
                })
                rows_written += 1
            else:
                print(f"Warning: Matching generation for {filename} (prefix {prefix}) not found.")

    print(f"CSV created at: {output_csv}")
    print(f"Total rows: {rows_written}")

if __name__ == "__main__":
    create_v04_infills_csv()

import csv
import os
from pathlib import Path

def create_v04_csv_flat():
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "synthetic_data" / "datasets" / "v04"
    renders_dir = dataset_dir / "renders"
    generations_dir = dataset_dir / "generations"
    output_csv = dataset_dir / "generation.csv"

    prompt_text = "Convert the input image to <isometric nyc pixel art>"

    if not renders_dir.exists():
        print(f"Error: Renders directory not found at {renders_dir}")
        return
    if not generations_dir.exists():
        print(f"Error: Generations directory not found at {generations_dir}")
        return

    generation_files = sorted([f.name for f in generations_dir.glob("*.png")])
    print(f"Found {len(generation_files)} generation files.")

    # Write CSV
    # User requested columns: "render", "generations", "prompt"
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['render', 'generations', 'prompt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        rows_written = 0
        for filename in generation_files:
            render_path = renders_dir / filename
            if render_path.exists():
                writer.writerow({
                    'render': f"renders/{filename}",
                    'generations': f"generations/{filename}",
                    'prompt': prompt_text
                })
                rows_written += 1
            else:
                print(f"Warning: Matching render for {filename} not found.")

    print(f"CSV created at: {output_csv}")
    print(f"Total rows: {rows_written}")

if __name__ == "__main__":
    create_v04_csv_flat()

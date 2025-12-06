import csv
import os
from pathlib import Path

def create_v04_generation_csv():
    project_root = Path(__file__).resolve().parents[2]
    # Update paths to match where the files were actually copied in the previous steps
    # Previous step created synthetic_data/datasets/v04/generation/renders and .../generations
    # BUT wait, checking the tool output from previous turns...
    
    # Turn 11 output: Target directory: .../synthetic_data/datasets/v04/generation/generations
    # Turn 13 output: Target directory: .../synthetic_data/datasets/v04/generation/renders
    
    dataset_dir = project_root / "synthetic_data" / "datasets" / "v04" / "generation"
    renders_dir = dataset_dir / "renders"
    generations_dir = dataset_dir / "generations"
    output_csv = dataset_dir / "generation.csv"

    prompt_text = "Convert the input image to <isometric nyc pixel art>"

    # Get file lists
    if not renders_dir.exists():
        print(f"Error: Renders directory not found at {renders_dir}")
        return
    if not generations_dir.exists():
        print(f"Error: Generations directory not found at {generations_dir}")
        return

    # Use generation directory as the source of truth for 'nnn' as per prompt
    # "for every nnn in @synthetic_data/datasets/v04/generations/**" (which maps to our nested path)
    generation_files = sorted([f.name for f in generations_dir.glob("*.png")])
    
    print(f"Found {len(generation_files)} generations.")
    
    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['render', 'generation', 'prompt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        rows_written = 0
        for filename in generation_files:
            # Check if corresponding render exists (good practice, though prompt implies iterating gens)
            render_path = renders_dir / filename
            if render_path.exists():
                # Paths should be relative to the CSV location? 
                # Usually datasets expect paths relative to the csv or project root.
                # Prompt examples: "renders/nnn.png", "generations/nnn.png"
                writer.writerow({
                    'render': f"renders/{filename}",
                    'generation': f"generations/{filename}",
                    'prompt': prompt_text
                })
                rows_written += 1
            else:
                print(f"Warning: Matching render for {filename} not found.")

    print(f"CSV created at: {output_csv}")
    print(f"Total rows: {rows_written}")

if __name__ == "__main__":
    create_v04_generation_csv()

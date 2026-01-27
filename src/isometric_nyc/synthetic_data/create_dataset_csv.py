import csv
import os
from pathlib import Path

def create_dataset_csv():
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "synthetic_data" / "datasets"
    v02_dir = dataset_dir / "v02"
    renders_dir = v02_dir / "renders"
    generations_dir = v02_dir / "generations"
    prompt_file = v02_dir / "generation_prompt.txt"
    output_csv = dataset_dir / "simple_generation.csv"

    # Read prompt
    if not prompt_file.exists():
        print(f"Error: Prompt file {prompt_file} not found.")
        return
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()

    # Get file lists
    if not renders_dir.exists() or not generations_dir.exists():
        print("Error: Renders or Generations directory not found.")
        return

    render_files = {f.name for f in renders_dir.glob("*.png")}
    generation_files = {f.name for f in generations_dir.glob("*.png")}

    # Intersection
    common_files = sorted(list(render_files.intersection(generation_files)))
    
    print(f"Found {len(render_files)} renders and {len(generation_files)} generations.")
    print(f"Creating CSV with {len(common_files)} paired entries.")

    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['render', 'generation', 'prompt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for filename in common_files:
            # Using v02/ prefix to match the actual directory structure relative to the CSV file
            writer.writerow({
                'render': f"v02/renders/{filename}",
                'generation': f"v02/generations/{filename}",
                'prompt': prompt_text
            })

    print(f"CSV created at: {output_csv}")

if __name__ == "__main__":
    create_dataset_csv()

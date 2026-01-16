"""
Merge label from second JSON file into first JSON file and apply pred_value correction.
Usage: python merge_label.py --input1 <input1.json> --input2 <input2.json> --output <output_dir>
"""

import json
import sys
import os
import argparse
from pathlib import Path

def merge_labels(input1_path, input2_path, output_dir):
    """
    Merge labels from input2 into input1, apply correction to pred_value, and save to output_dir/results.json
    
    Args:
        input1_path: Path to first JSON file (affinity_predictions.json)
        input2_path: Path to second JSON file (results.json with labels)
        output_dir: Directory to save merged results
    """
    
    # Read first JSON file
    print(f"Reading first JSON file: {input1_path}")
    with open(input1_path, 'r') as f:
        data1 = json.load(f)
    
    # Read second JSON file
    print(f"Reading second JSON file: {input2_path}")
    with open(input2_path, 'r') as f:
        data2 = json.load(f)
    
    # Merge labels and apply correction
    print("Merging labels and applying pred_value correction...")
    merged_count = 0
    missing_count = 0
    corrected_count = 0
    
    for key in data1:
        if key in data2:
            # Add label to the first file's value
            data1[key].update(data2[key])
            merged_count += 1
        else:
            print(f"Warning: Key '{key}' not found in second file or missing 'label' field")
            missing_count += 1
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save merged results
    output_file = output_path / "results.json"
    print(f"Saving merged results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data1, f, indent=2)
    
    print(f"Merge and correction completed!")
    print(f"Successfully merged labels for {merged_count} keys")
    print(f"Applied pred_value correction for {corrected_count} keys")
    print(f"Missing labels for {missing_count} keys")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge labels from second JSON file into first JSON file and apply pred_value correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_label.py --input1 affinity_predictions.json --input2 results.json --output ./output
  python merge_label.py -i1 /path/to/affinity_predictions.json -i2 /path/to/results.json -o ./output
        """
    )
    
    parser.add_argument(
        '--input1', '-i1',
        help='Path to first JSON file (affinity_predictions.json)'
    )
    
    parser.add_argument(
        '--input2', '-i2',
        help='Path to second JSON file (results.json with labels)'
    )
    
    parser.add_argument(
        '--outdir', '-o',
        help='Directory to save merged results'
    )
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.input1):
        print(f"Error: First input file not found: {args.input1}")
        sys.exit(1)
    
    if not os.path.exists(args.input2):
        print(f"Error: Second input file not found: {args.input2}")
        sys.exit(1)
    
    try:
        merge_labels(args.input1, args.input2, args.outdir)
    except Exception as e:
        print(f"Error during merge: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

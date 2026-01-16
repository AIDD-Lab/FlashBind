#!/usr/bin/env python3
"""
Evaluation script for bioassay results.
Evaluates binary classification performance using 'binary' field as scores and 'label' field as true labels.
Calculates metrics for one or more input files and reports the average.
"""

import os
import argparse
import json
from tqdm import tqdm
import sys
import numpy as np

sys.path.insert(0, "./src")
from affinity.utils.metrics import (
    calculate_metrics_per_target,
    calculate_global_auroc,
    calculate_global_auprc,
    save_json_with_nan_handling,
)


def extract_binary_scores_from_json(json_file_path, score_field="enzyme", target_dataset="all"):
    """
    Extract binary classification scores and labels from JSON file.
    Args:
        json_file_path (str): Path to the JSON file
        score_field (str): Field name for predicted probabilities
        target_dataset (str): Dataset prefix to filter (default: "all")
    Returns:
        dict: Dictionary containing extracted scores and labels for each entry
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    results = {}
    for entry_id, entry_data in data.items():
        if target_dataset != "all":
            if not entry_id.startswith(target_dataset):
                continue

        if score_field not in entry_data or 'label' not in entry_data:
            continue
        results[entry_id] = {
            'score': entry_data[score_field],
            'label': entry_data['label'],
        }
    return results

def print_binary_scores_summary(scores_data):
    """
    Print a summary of the extracted binary classification scores.
    Args:
        scores_data (dict): Dictionary containing extracted scores
    """
    print("Binary classification score extraction summary:")
    print(f"Total entries processed: {len(scores_data)}")
    
    if scores_data:
        labels = [entry_data['label'] for entry_data in scores_data.values()]
        scores = [entry_data['score'] for entry_data in scores_data.values()]
        
        # Count label distribution (assuming binary labels)
        label_counts = {}
        for label in labels:
            if isinstance(label, str):
                key = label
            else:
                key = 1 if label >= 0.5 else 0
            label_counts[key] = label_counts.get(key, 0) + 1
        
        print(f"Label distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
        
        print(f"Score statistics:")
        print(f"  Min: {min(scores):.6f}")
        print(f"  Max: {max(scores):.6f}")
        print(f"  Mean: {sum(scores)/len(scores):.6f}")
        
        first_entry = next(iter(scores_data.keys()))
        print(f"\nExample entry ({first_entry}):")
        print(f"  Label: {scores_data[first_entry]['label']}")
        print(f"  Score: {scores_data[first_entry]['score']:.6f}")


def main():
    """
    Main function to evaluate bioassay results.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate bioassay results using binary scores and labels from one or more files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single file
  python eval.py --input results.json --output ./metrics.json

  # Evaluate multiple files and get the average
  python eval.py -i results1.json results2.json results3.json -o ./metrics.json
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        nargs='+', 
        help='Path(s) to JSON file(s) containing results with binary scores and labels'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path for average metrics (default: ./metrics.json)'
    )
    
    parser.add_argument(
        '--score_field', '-s',
        default='enzyme',
        help='Field name for predicted probabilities in the JSON file (default: affinity_probability_binary)'
    )

    parser.add_argument(
        '--dataset', '-d',
        default='all',
        # default='Esterase',
        help='Dataset prefix to filter keys (e.g. "enzyme"). Default is "all".'
    )
    
    args = parser.parse_args()
    
    all_aurocs = []
    all_auprcs = []
    individual_results = {}

    print(f"Processing {len(args.input)} input file(s)...")

    for input_file in args.input:
        print(f"\n--- Processing file: {input_file} ---")
        
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} not found! Skipping.")
            continue
        
        try:
            print(f"Extracting scores from JSON file using field '{args.score_field}' (Dataset filter: {args.dataset})...")
            scores_data = extract_binary_scores_from_json(input_file, args.score_field, args.dataset)
            
            if not scores_data:
                print(f"No data extracted from this file (matching dataset: {args.dataset}). Skipping.")
                continue
                
            print_binary_scores_summary(scores_data)
            
            print("Calculating global AUROC...")
            global_auroc = calculate_global_auroc(scores_data)
            
            print("Calculating global AUPRC...")
            global_auprc = calculate_global_auprc(scores_data)
            
            if global_auroc is not None and not np.isnan(global_auroc):
                all_aurocs.append(global_auroc)
            if global_auprc is not None and not np.isnan(global_auprc):
                all_auprcs.append(global_auprc)
            
            individual_results[input_file] = {
                'AUROC': global_auroc,
                'AUPRC': global_auprc,
                'entries': len(scores_data)
            }
            
            print(f"Metrics for {input_file}:")
            print(f"  AUROC: {global_auroc:.6f}")
            print(f"  AUPRC: {global_auprc:.6f}")
            
        except Exception as e:
            print(f"Error processing file {input_file}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_aurocs or not all_auprcs:
        print("\nNo valid metrics calculated from any file. Exiting.")
        return None

    avg_auroc = sum(all_aurocs) / len(all_aurocs)
    avg_auprc = sum(all_auprcs) / len(all_auprcs)
    
    results = {
        'average_global_AUROC': avg_auroc,
        'average_global_AUPRC': avg_auprc,
        'summary': {
            'total_files_processed_successfully': len(all_aurocs),
            'input_files_attempted': args.input,
        },
        'individual_results': individual_results,
    }
    
    save_json_with_nan_handling(results, args.output)
    print(f"\nAverage metrics results saved to {args.output}")
    
    print("\n=== OVERALL EVALUATION SUMMARY (AVERAGES) ===")
    print(f"Successfully processed {len(all_aurocs)} file(s).")
    print(f"Average AUROC: {avg_auroc:.6f}")
    print(f"Average AUPRC: {avg_auprc:.6f}")
    
    return results


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Evaluation script for bioassay results.
Evaluates binary classification performance using 'binary' field as scores and 'label' field as true labels.
"""

import os
import argparse
import json
from tqdm import tqdm
import sys
sys.path.insert(0, "./src")
from affinity.utils.metrics import (
    calculate_metrics_per_target,
    calculate_global_auroc,
    calculate_global_auprc,
    save_json_with_nan_handling,
)

def extract_binary_scores_from_json(json_file_path, score_field="affinity_probability_binary"):
    """
    Extract binary classification scores and labels from JSON file.
    Args:
        json_file_path (str): Path to the JSON file
        score_field (str): Field name for predicted probabilities
    Returns:
        dict: Dictionary containing extracted scores and labels for each entry
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    results = {}
    for entry_id, entry_data in data.items():
        if score_field not in entry_data or 'label' not in entry_data:
            continue
        results[entry_id] = {
            'label': entry_data['label'],
            'score': entry_data[score_field]
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
        description="Evaluate bioassay results using binary scores and labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py --input results.json --output ./metrics.json
  python eval.py -i results.json -o ./metrics.json
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        help='Path to JSON file containing results with binary scores and labels'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path for metrics (default: ./metrics.json)'
    )
    
    parser.add_argument(
        '--score-field', '-s',
        default='binary',
        help='Field name for predic,ted probabilities in the JSON file (default: affinity_probability_binary)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found!")
        return
    
    try:
        # Extract scores
        print(f"Extracting scores from JSON file using field '{args.score_field}'...")
        scores_data = extract_binary_scores_from_json(args.input, args.score_field)
        
        # Print summary
        print_binary_scores_summary(scores_data)
        
        # Calculate per-target metrics
        print("\nCalculating per-target metrics...")
        avg_metrics, target_metrics = calculate_metrics_per_target(scores_data)
        
        # Calculate global AUROC
        print("Calculating global AUROC...")
        global_auroc = calculate_global_auroc(scores_data)
        
        print("Calculating global AUPRC...")
        global_auprc = calculate_global_auprc(scores_data)
        
        # Prepare final results
        results = {
            'average_metrics': avg_metrics,
            'global_AUROC': global_auroc,
            'global_AUPRC': global_auprc,
            'per_target_metrics': target_metrics,
            'summary': {
                'total_entries': len(scores_data),
                'total_targets': len(target_metrics)
            }
        }
        
        # Save results
        save_json_with_nan_handling(results, args.output)
        print(f"\nMetrics results saved to {args.output}")
        
        # Print summary
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Average AP: {avg_metrics['AP']:.6f}")
        print(f"Average AUROC: {avg_metrics['AUROC']:.6f}")
        print(f"Average AUPRC: {avg_metrics['AUPRC']:.6f}")
        print(f"Global AUROC: {global_auroc:.6f}")
        print(f"Global AUPRC: {global_auprc:.6f}")
        print(f"EF_0.5: {avg_metrics['EF_0_5']:.6f}")
        print(f"EF_1: {avg_metrics['EF_1']:.6f}")
        print(f"EF_2: {avg_metrics['EF_2']:.6f}")
        print(f"EF_5: {avg_metrics['EF_5']:.6f}")
        
        return results
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()

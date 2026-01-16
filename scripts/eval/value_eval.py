#!/usr/bin/env python3
"""
Evaluation script for bioassay value regression results.
Evaluates regression performance using predicted values and 'label' field as true values.
All calculations are performed in kcal/mol units after converting from log10(uM).
"""

import os
import argparse
import json
from tqdm import tqdm
import sys
sys.path.insert(0, "./src")
from affinity.utils.metrics import (
    calculate_value_metrics_per_target,
    save_json_with_nan_handling,
)

def extract_value_scores_from_json(json_file_path, score_field="affinity_pred_value"):
    """
    Extract value regression scores and labels from JSON file.
    Args:
        json_file_path (str): Path to the JSON file
        score_field (str): Field name for predicted scores
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


def print_value_scores_summary(scores_data):
    """
    Print a summary of the extracted value regression scores.
    Args:
        scores_data (dict): Dictionary containing extracted scores
    """
    print("Value regression score extraction summary:")
    print(f"Total entries processed: {len(scores_data)}")
    
    if scores_data:
        labels = [entry_data['label'] for entry_data in scores_data.values()]
        scores = [entry_data['score'] for entry_data in scores_data.values()]
        
        print(f"Label statistics:")
        print(f"  Min: {min(labels):.6f}")
        print(f"  Max: {max(labels):.6f}")
        print(f"  Mean: {sum(labels)/len(labels):.6f}")
        
        print(f"Score statistics:")
        print(f"  Min: {min(scores):.6f}")
        print(f"  Max: {max(scores):.6f}")
        print(f"  Mean: {sum(scores)/len(scores):.6f}")
        
        first_entry = next(iter(scores_data.keys()))
        print(f"\nExample entry ({first_entry}):")
        print(f"  Label: {scores_data[first_entry]['label']:.6f}")
        print(f"  Score: {scores_data[first_entry]['score']:.6f}")


def main():
    """
    Main function to evaluate bioassay value regression results.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate bioassay value regression results using predicted and true log10(uM) values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python value_eval.py --input results.json --output ./value_metrics.json
  python value_eval.py -i results.json -o ./value_metrics.json
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        help='Path to JSON file containing results with predicted and true log10(uM) values'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path for metrics (default: ./value_metrics.json)'
    )
    
    parser.add_argument(
        '--score-field', '-s',
        default='pred_value',
        help='Field name for predicted scores in the JSON file (default: affinity_pred_value)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found!")
        return
    
    try:
        # Extract scores
        print(f"Extracting scores from JSON file using field '{args.score_field}'...")
        scores_data = extract_value_scores_from_json(args.input, args.score_field)
        
        # Print summary
        print_value_scores_summary(scores_data)
        
        # Calculate per-target value regression metrics
        print("\nCalculating per-target value regression metrics...")
        avg_metrics, target_metrics = calculate_value_metrics_per_target(scores_data)
        
        # Prepare final results
        results = {
            'average_metrics': avg_metrics,
            'per_target_metrics': target_metrics,
            'summary': {
                'total_entries': len(scores_data),
                'total_targets': len(target_metrics)
            }
        }
        
        # Save results
        save_json_with_nan_handling(results, args.output)
        print(f"\nValue regression metrics results saved to {args.output}")
        
        # Print summary
        print("\n=== VALUE REGRESSION EVALUATION SUMMARY ===")
        print(f"Pearson R: {avg_metrics['Pearson_R']:.6f}")
        print(f"Kendall Tau: {avg_metrics['Kendall_Tau']:.6f}")
        print(f"PMAE: {avg_metrics['PMAE']:.6f} kcal/mol")
        print(f"MAE: {avg_metrics['MAE']:.6f} kcal/mol")
        print(f"MAE (centered): {avg_metrics['MAE_cent']:.6f} kcal/mol")
        print(f"Percentage within 1 kcal/mol: {avg_metrics['Perc_1kcal']:.2f}%")
        print(f"Percentage within 2 kcal/mol: {avg_metrics['Perc_2kcal']:.2f}%")
        print(f"Percentage within 1 kcal/mol (centered): {avg_metrics['Perc_1kcal_cent']:.2f}%")
        print(f"Percentage within 2 kcal/mol (centered): {avg_metrics['Perc_2kcal_cent']:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()

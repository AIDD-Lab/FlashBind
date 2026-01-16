"""
Ensemble Script for Affinity Predictions

This script combines multiple model prediction results into ensemble predictions
by averaging pred_value and binary scores across models.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import sys

@rank_zero_only
def create_ensemble_results(all_results: List[Dict[str, Any]], output_dir: Path, task: str = "binary") -> None:
    """Create ensemble results with strict success criteria."""
    
    TASK_KEYS = {
        "binary": ["binary"],
        "value": ["pred_value", "pred_value_raw", "mw"],
        "enzyme": ["enzyme"],
    }
    keys_to_keep = TASK_KEYS.get(task, [])
    
    ensemble_results = {}
    all_record_ids = set()
    for results in all_results:
        all_record_ids.update(results.keys())
    
    for record_id in all_record_ids:
        values_by_key = {k: [] for k in keys_to_keep}
        all_success = True
        n_models_with_record = 0
        
        for results in all_results:
            if record_id in results:
                n_models_with_record += 1
                record_result = results[record_id]
                if record_result.get('status') == 'success':
                    for k in keys_to_keep:
                        if record_result.get(k) is not None:
                            values_by_key[k].append(record_result[k])
                else:
                    all_success = False
                    break
            else:
                all_success = False
                break
        
        has_values = any(values_by_key[k] for k in keys_to_keep)
        if all_success and n_models_with_record == len(all_results) and has_values:
            result = {'status': 'success', 'n_models': n_models_with_record}
            for k in keys_to_keep:
                result[k] = float(np.mean(values_by_key[k])) if values_by_key[k] else None
        else:
            result = {'status': 'failed', 'n_models': n_models_with_record, 'failure_reason': 'not_all_models_successful'}
            for k in keys_to_keep:
                result[k] = None
        ensemble_results[record_id] = result
    
    # Save ensemble results
    ensemble_file = output_dir / "affinity_predictions_ensemble.json"
    with ensemble_file.open('w') as f:
        json.dump(ensemble_results, f, indent=2)
    
    print(f"\nEnsemble results saved to: {ensemble_file}")
    
    # Print statistics
    total_records = len(ensemble_results)
    successful_records = sum(1 for r in ensemble_results.values() if r.get('status') == 'success')
    print(f"Ensemble Summary:")
    print(f"  Total records: {total_records}")
    print(f"  Successful: {successful_records}, Failed: {total_records - successful_records}")
    
    if successful_records > 0:
        for k in keys_to_keep:
            vals = [r[k] for r in ensemble_results.values() if r.get('status') == 'success' and r.get(k) is not None]
            if vals:
                print(f"  {k} stats: min={np.min(vals):.4f}, max={np.max(vals):.4f}, mean={np.mean(vals):.4f}")

def load_prediction_results(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Load prediction results from multiple JSON files.
    
    Parameters
    ----------
    file_paths : List[str]
        List of paths to JSON files containing prediction results
        
    Returns
    -------
    List[Dict[str, Any]]
        List of prediction result dictionaries, one per model
        
    Raises
    ------
    FileNotFoundError
        If any input file doesn't exist
    json.JSONDecodeError
        If any file contains invalid JSON
    """
    all_results = []
    
    for i, file_path in enumerate(file_paths):
        file_path = Path(file_path).expanduser()
        
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        try:
            print(f"Loading model {i+1} results from: {file_path}")
            with file_path.open('r') as f:
                results = json.load(f)
            
            # Validate that it's a dictionary
            if not isinstance(results, dict):
                raise ValueError(f"Expected dictionary format in {file_path}, got {type(results)}")
            
            all_results.append(results)
            print(f"  Loaded {len(results)} records from model {i+1}")
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")
        except Exception as e:
            raise Exception(f"Error loading file {file_path}: {e}")
    
    return all_results


def main():
    """Main function to run ensemble prediction."""
    parser = argparse.ArgumentParser(
        description="Create ensemble predictions from multiple model results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input_files",
        type=str,
        nargs='+',
        help="Paths to JSON files containing prediction results from different models",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save ensemble results",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.input_files) < 2:
        raise ValueError("At least 2 input files are required for ensemble")
    
    # Create output directory
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating ensemble from {len(args.input_files)} models...")
    print(f"Output directory: {output_dir}")
    
    try:
        # Load all prediction results
        all_results = load_prediction_results(args.input_files)
        
        # Create ensemble results using the existing function
        print(f"\n{'='*60}")
        print("Generating ensemble results...")
        print(f"{'='*60}")
        create_ensemble_results(all_results, output_dir)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Affinity Prediction Script
"""

import argparse
import platform
import warnings
import sys
import time
from pathlib import Path
import json
from typing import List, Union, Dict, Any
from pytorch_lightning.utilities import rank_zero_only
import torch
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from typing import Optional
sys.path.insert(0, "./src")

from affinity.model.model import AffinityModel
from affinity.dataset.inference import (
    AffinityInferenceDataset,
    AffinityInferenceDataModule,
)
from affinity.utils.writer import AffinityPredictionWriter

def preprocess_param_value(key: str, value: Any) -> Any:
    """Preprocess parameter values to handle special cases.
    
    Parameters
    ----------
    key : str
        Parameter name
    value : Any
        Parameter value
        
    Returns
    -------
    Any
        Preprocessed parameter value
    """
    if key == 'disable_ema':
        if value is None:
            return False
        elif isinstance(value, list) and len(value) == 0:
            return False
        elif isinstance(value, list):
            return [str(v).lower() in ['true', '1', 'yes'] for v in value]
        else:
            return str(value).lower() in ['true', '1', 'yes']
    
    # Other parameters returned as-is
    return value


def apply_broadcast_logic(value: Any, n_models: int, param_name: str) -> List[Any]:
    """Apply unified broadcast logic for all parameters.
    
    Parameters
    ----------
    value : Any
        Preprocessed parameter value
    n_models : int
        Number of models
    param_name : str
        Parameter name for error messages
        
    Returns
    -------
    List[Any]
        List of parameter values matching model count
    """
    if value is None:
        return [None] * n_models
    elif isinstance(value, list):
        if len(value) == 1:
            # Single element list, broadcast to all models
            return value * n_models
        elif len(value) == n_models:
            # Length matches, use directly
            return value
        else:
            raise ValueError(f"Parameter {param_name} list length ({len(value)}) must be 1 or {n_models}")
    else:
        # Single value, broadcast to all models
        return [value] * n_models


def normalize_params(checkpoints: Union[str, List[str]], **params) -> Dict[str, List[Any]]:
    """Normalize parameters to lists matching checkpoint count.
    
    Parameters
    ----------
    checkpoints : Union[str, List[str]]
        Single checkpoint path or list of checkpoint paths
    **params : Dict[str, Any]
        Parameters that may be single values or lists
        
    Returns
    -------
    Dict[str, List[Any]]
        Normalized parameters with all values as lists
    """
    # Convert single checkpoint to list
    if isinstance(checkpoints, str):
        checkpoints = [checkpoints]
    
    n_models = len(checkpoints)
    normalized = {'checkpoints': checkpoints}
    
    for key, value in params.items():
        # Preprocess parameter value to handle special cases
        processed_value = preprocess_param_value(key, value)
        
        # Apply unified broadcast logic
        normalized[key] = apply_broadcast_logic(processed_value, n_models, key)
    
    return normalized

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

def main(args) -> None:
    """Run affinity prediction with simplified inference inputs."""
    # Environment setup
    warnings.filterwarnings("ignore", ".*that has Tensor Cores.*")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    if args.seed is not None:
        seed_everything(args.seed)

    # Resolve paths
    id_list_path = Path(args.data).expanduser()
    out_dir = Path(args.out_dir).expanduser() / f"affinity_results_{id_list_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize parameters for multi-model support
    normalized_params = normalize_params(
        args.affinity_checkpoint,
        protein_repr=args.protein_repr,
        ligand_repr=args.ligand_repr,
        morgan_repr=args.morgan_repr,
        unimol_repr=args.unimol_repr,
        pocket_indices=args.pocket_indices,
        distance_threshold=args.distance_threshold,
        disable_ema=args.disable_ema
    )
    
    checkpoints = normalized_params['checkpoints']
    need_ensemble = len(checkpoints) > 1
    all_results = [] if need_ensemble else None

    print(f"Processing {len(checkpoints)} model(s)...")
    if need_ensemble:
        print("Ensemble mode enabled - will generate ensemble results after all models complete.")
    # Process each model
    for i, checkpoint in enumerate(checkpoints):
        print(f"\n{'='*60}")
        print(f"Processing model {i+1}/{len(checkpoints)}: {checkpoint}")
        print(f"{'='*60}")
        
        # Build dataset with current model's parameters
        dataset = AffinityInferenceDataset.from_config(
            structure=args.structure,
            structure_type=args.structure_type,
            ligand=args.ligand,
            ligand_type=args.ligand_type,
            protein_repr=normalized_params['protein_repr'][i],
            ligand_repr=normalized_params['ligand_repr'][i],
            morgan_repr=normalized_params['morgan_repr'][i],
            unimol_repr=normalized_params['unimol_repr'][i],
            id_list=args.data,
            pocket_indices=normalized_params['pocket_indices'][i],
            distance_threshold=normalized_params['distance_threshold'][i],
        )

        data_module = AffinityInferenceDataModule(
            datasets=[dataset],
            num_workers=args.num_workers,
        )

        # Determine output filename
        if need_ensemble:
            output_filename = f"affinity_predictions_{i}"
        else:
            output_filename = "affinity_predictions"
        
        # Writer
        pred_writer = AffinityPredictionWriter(
            output_dir=str(out_dir),
            output_filename=output_filename,
            task=args.task,
        )

        # Trainer setup
        strategy = "auto"
        if args.devices > 1:
            strategy = DDPStrategy()
            print(f"Using {args.devices} devices, DDP strategy")

        # Load model
        print(f"Loading affinity model from: {checkpoint}")

        raw_ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = raw_ckpt["state_dict"]
        hparams = raw_ckpt["hyper_parameters"].copy()

        disable_ema_for_this_model = normalized_params['disable_ema'][i]

        if not disable_ema_for_this_model and "ema" in raw_ckpt and raw_ckpt["ema"].get("ema_weights") is not None:
            print(f"  Using EMA weights for model {i+1}")
            state_dict = raw_ckpt["ema"]["ema_weights"]
        elif disable_ema_for_this_model:
            print(f"  Disabling EMA for model {i+1}, using original weights")
        else:
            print(f"  No EMA weights found, using original weights")

        hparams['ema'] = False

        keys_to_remove = []
        if args.task == 'enzyme':
            keys_to_remove = [k for k in state_dict.keys() if "value_predictor" in k or "prob_predictor" in k]
        elif args.task == 'value':
            keys_to_remove = [k for k in state_dict.keys() if "enzyme_predictor" in k or "prob_predictor" in k]
        elif args.task == 'binary':
            keys_to_remove = [k for k in state_dict.keys() if "value_predictor" in k or "enzyme_predictor" in k]
            
        if keys_to_remove:
            print(f"Task '{args.task}': Filtering out {len(keys_to_remove)} mismatched keys (e.g., {keys_to_remove[0]})")
            for k in keys_to_remove:
                del state_dict[k]

        model_module = AffinityModel(**hparams,)
        model_module.load_state_dict(state_dict, strict=False)
        model_module.eval()

        trainer = Trainer(
            default_root_dir=str(out_dir),
            strategy=strategy,
            callbacks=[pred_writer],
            accelerator=args.accelerator,
            devices=args.devices,
            precision=32,
        )

        # Predict
        trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=False,
        )
        
        # If ensemble mode, collect results for later processing
        if need_ensemble:
            results_file = out_dir / f"{output_filename}.json"
            if results_file.exists():
                try:
                    with results_file.open('r') as f:
                        model_results = json.load(f)
                    all_results.append(model_results)
                    print(f"Collected results from model {i+1}: {len(model_results)} records")
                except Exception as e:
                    print(f"Warning: Failed to load results from {results_file}: {e}")
            else:
                print(f"Warning: Results file not found: {results_file}")
        
        # Wait 5 seconds between models to allow resource cleanup
        if i < len(checkpoints) - 1:  # Don't wait after the last model
            print(f"Waiting 5 seconds before processing next model...")
            time.sleep(5)
    
    # Generate ensemble results if multiple models were used
    if need_ensemble and all_results:
        print(f"\n{'='*60}")
        print("Generating ensemble results...")
        print(f"{'='*60}")
        create_ensemble_results(all_results, out_dir, args.task)
    elif need_ensemble:
        print("Warning: No results collected for ensemble generation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bioassay Affinity Prediction with simplified inference inputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required inputs for the new inference pipeline
    parser.add_argument(
        "--data",
        type=str,
        default="./data/mf-pcba/id.json",
        help="Path to a json file with one sample id per line (format: {proteinId}_{ligandId})",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="binary",
        choices=["value", "binary", "enzyme"],
        help="Specify the task to filter model weights. If 'enzyme', value/binary predictors will be ignored to avoid shape mismatch.",
    )

    # Structure and representations sources
    parser.add_argument(
        "--structure",
        type=str,
        default="./data/mf-pcba/pdb",
        help="Structure source (directory or LMDB) for protein structures",
    )
    parser.add_argument(
        "--structure_type",
        type=str,
        default="pdb",
        help="Structure type (pdb or cif)",
    )
    parser.add_argument(
        "--ligand",
        type=str,
        default="./data/mf-pcba/ligand_sdf.lmdb",
        help="SDF source (directory or LMDB) for ligand structures",
    )
    parser.add_argument(
        "--ligand_type",
        type=str,
        default="sdf",
        help="Ligand type (sdf or smils)",
    )
    parser.add_argument(
        "--protein_repr",
        type=str,
        nargs='+',
        default="./data/mf-pcba/repr/esm3.pt",
        help="Protein representation source(s) (file or directory). Single value will be used for all models.",
    )
    parser.add_argument(
        "--ligand_repr",
        type=str,
        nargs='*',
        default="./data/mf-pcba/repr/torchdrug.lmdb",
        help="Ligand representation source(s) (file or LMDB). Single value will be used for all models.",
    )
    parser.add_argument(
        "--morgan_repr",
        type=str,
        nargs='*',
        default=None,
        help="Optional Morgan fingerprint representation source(s) (file or LMDB). Single value will be used for all models.",
    )
    parser.add_argument(
        "--unimol_repr",
        type=str,
        nargs='*',
        default=None,
        help="Optional UniMol representation source(s) (file or LMDB). Single value will be used for all models.",
    )
    parser.add_argument(
        "--pocket_indices",
        type=str,
        nargs='*',
        default=None,
        help="Optional pocket residue indices source(s) aligned with sample ids. Single value will be used for all models.",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        nargs='+',
        default=20.0,
        help="Distance threshold(s) for selecting pocket residues. Single value will be used for all models.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./",
        help="Directory to save predictions (a subfolder per id list will be created)",
    )
    parser.add_argument(
        "--affinity_checkpoint",
        type=str,
        nargs='+',
        default=["./checkpoints/v1/binary_1.ckpt", "./checkpoints/v1/binary_2.ckpt"],
        help="Path(s) to the trained affinity model checkpoint(s). Can be a single path or multiple paths for ensemble.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use for prediction",
    )
    parser.add_argument(
        "--accelerator",
        choices=["gpu", "cpu", "tpu"],
        default="gpu",
        help="Accelerator for prediction",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed; inference is deterministic by design",
    )
    parser.add_argument(
        "--disable_ema",
        nargs='*',
        default=False,
        help="Disable EMA during inference. Use '--disable_ema' to disable for all models, or '--disable_ema True False' to specify per model.",
    )

    args = parser.parse_args()
    main(args)
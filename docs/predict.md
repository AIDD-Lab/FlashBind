# Prediction Guide

This document describes how to run inference with BioAssay models.

## Quick Start

We provide four prediction scripts in the root directory to reproduce the results in our paper:

| Script | Task | Description |
|--------|------|-------------|
| `predict_binary.sh` | Binary | Protein-ligand binary activity prediction |
| `predict_value.sh` | Value | Affinity value prediction |
| `predict_enzyme.sh` | Enzyme | Enzyme-substrate interaction prediction |
| `predict_antibiotic.sh` | Binary | Antibiotic activity prediction |

You can refer to these scripts and modify the parameters as needed.

## Ensemble Prediction

We support ensemble prediction by specifying multiple checkpoint paths. The final ensemble result will be the average of all model predictions.

```bash
CHECKPOINTS=(
    "./checkpoints/binary_1.ckpt"
    "./checkpoints/binary_2.ckpt"
)

python ./scripts/predict.py \
  --affinity_checkpoint "${CHECKPOINTS[@]}" \
  # ... other arguments
```

When using ensemble mode, the output directory will contain:
- `affinity_predictions_0.json`: Results from model 1
- `affinity_predictions_1.json`: Results from model 2
- `affinity_predictions_ensemble.json`: Averaged ensemble results

## Output Format

The prediction output is a JSON file with the following structure:

```json
{
  "{prot_id}_{ligand_id}": {
    "status": "success",
    "binary": 0.85
  }
}
```

### Fields

| Field | Task | Description |
|-------|------|-------------|
| `status` | All | `"success"` or `"failed"`, indicating whether the prediction succeeded |
| `binary` | Binary | Probability value (0~1) for binary activity prediction |
| `enzyme` | Enzyme | Probability value (0~1) for enzyme-substrate interaction prediction |
| `pred_value` | Value | Calibrated affinity value, log₁₀(IC₅₀/Ki/Kd) in μM units |
| `pred_value_raw` | Value | Raw predicted value before calibration |
| `mw` | Value | Molecular weight of the ligand |

> **Note**: Different tasks output different fields. For the value task, `pred_value` is calibrated based on `pred_value_raw`.

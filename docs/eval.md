# Evaluation Guide

This document describes how to evaluate model predictions.

## Label Format

You need to provide a `labels.json` file with the following format:

```json
{
    "{prot_id}_{ligand_id}": {
        "label": "True"
    },
    "{prot_id}_{ligand_id}": {
        "label": "False"
    },
    "{prot_id}_{ligand_id}": {
        "label": 1.0
    }
}
```

- For `binary` and `enzyme` tasks: label is `"True"` or `"False"`
- For `value` task: label is a float value (log₁₀ of the affinity value in μM)

## Evaluation Pipeline

### 1. Merge Predictions with Labels

First, merge the prediction results with labels:

```bash
python scripts/eval/merge_label.py \
    --input1 /path/to/affinity_predictions.json \
    --label_file /path/to/labels.json \
    --output_file /path/to/merged.json
```

### 2. Calculate Metrics

Then calculate metrics using the corresponding evaluation script:

```bash
# For binary task
python scripts/eval/binary_eval.py --input_file /path/to/merged.json

# For value task
python scripts/eval/value_eval.py --input_file /path/to/merged.json

# For enzyme task (supports multiple files for cross-validation)
python scripts/eval/enzyme_eval.py --input_files /path/to/fold1.json /path/to/fold2.json ...
```

> **Note**: For the enzyme task, we support multiple input files to calculate metrics under cross-validation settings.

## Ensemble

We provide `ensemble.py` to combine predictions from multiple models:

```bash
python scripts/eval/ensemble.py \
    --input_files /path/to/pred1.json /path/to/pred2.json ... \
    --output_dir /path/to/output/
```

The ensemble result is the average of predictions from all models.
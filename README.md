# FlashAffinity

[![Paper](https://img.shields.io/badge/Paper-MLSB%202025-blue)](https://openreview.net/pdf?id=TZTahjQNjX)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **"FlashAffinity: Bridging the Accuracy-Speed Gap in Protein-Ligand Binding Affinity Prediction"**, Accepted at **MLSB Workshop 2025**.

A ultra-fast protein-ligand affinity prediction framework supporting multiple prediction tasks: binary activity classification, enzyme-substrate interaction prediction, and affinity value regression.

## Environment Setup

```bash
conda env create -f env.yaml
conda activate flashaffinity
```

## Model & Data

- **Model Checkpoints**: [Model](https://huggingface.co/clorf6/FlashAffinity)
- **Datasets**: [Datasets](https://huggingface.co/datasets/clorf6/FlashAffinity)

## Supported Tasks

| Task | Description | Output |
|------|-------------|--------|
| `binary` | Binary activity classification | Activity probability (0-1) |
| `value` | Affinity value regression | log₁₀(IC50/Ki/Kd) value |
| `enzyme` | Enzyme-substrate interaction | Activity score (0-1) |

## Documentation

- [Training Guide](docs/train.md)
- [Prediction Guide](docs/predict.md)
- [Data Processing Guide](docs/data_process.md)
- [Evaluation Guide](docs/eval.md)

## Citation

```bibtex
@article {Jiang2025.12.22.695983,
	author = {Jiang, Songlin and Chen, Yifan and Cao, Ze and Jin, Wengong},
	title = {FlashAffinity: Bridging the Accuracy-Speed Gap in Protein-Ligand Binding Affinity Prediction},
	year = {2025},
	journal = {bioRxiv}
}
```

## Acknowledgments

We thank the [FABind+](https://github.com/QizhiPei/FABind) team and [Boltz-2](https://github.com/jwohlwend/boltz) team for their outstanding contributions to the community and for making their code openly available.
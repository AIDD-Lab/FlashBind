#!/bin/bash

CHECKPOINTS=(
    "./checkpoints/binary_1.ckpt"
    "./checkpoints/binary_2.ckpt"
)

torchrun --nproc_per_node=4 --rdzv_endpoint="localhost:29546" ./scripts/predict.py \
--data ./data/mf-pcba/subset_id.json \
--structure ./data/mf-pcba/pdb \
--structure_type pdb \
--ligand ./data/mf-pcba/ligand_sdf.lmdb \
--ligand_type sdf \
--pocket_indices ./data/mf-pcba/pocket_indices.lmdb \
--protein_repr ./data/mf-pcba/repr/esm3.pt \
--ligand_repr ./data/mf-pcba/repr/torchdrug.lmdb \
--distance_threshold 20.0 \
--out_dir ./binary \
--devices 4 \
--affinity_checkpoint "${CHECKPOINTS[@]}"

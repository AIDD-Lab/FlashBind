#!/bin/bash

CHECKPOINTS=(
    "./checkpoints/binary_1.ckpt"
    "./checkpoints/binary_2.ckpt"
)

torchrun --nproc_per_node=4 --rdzv_endpoint="localhost:29546" ./scripts/predict.py \
--data ./data/antibiotic/id.json \
--structure ./data/antibiotic/pdb \
--structure_type pdb \
--ligand ./data/antibiotic/ligand_sdf.lmdb \
--ligand_type sdf \
--protein_repr ./data/antibiotic/repr/esm3.lmdb \
--ligand_repr ./data/antibiotic/repr/torchdrug.lmdb \
--distance_threshold 20.0 \
--out_dir ./antibiotic \
--devices 4 \
--affinity_checkpoint "${CHECKPOINTS[@]}"

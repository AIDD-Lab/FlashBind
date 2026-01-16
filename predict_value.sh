#!/bin/bash

CHECKPOINTS=(
    "./checkpoints/value_1.ckpt"
    "./checkpoints/value_2.ckpt"
)

torchrun --nproc_per_node=4 --rdzv_endpoint="localhost:29501" ./scripts/predict.py \
--data ./data/openfe/id.json \
--task value \
--structure ./data/openfe/pdb \
--structure_type pdb \
--ligand ./data/openfe/ligand_sdf.lmdb \
--ligand_type sdf \
--protein_repr ./data/openfe/repr/esm3.lmdb \
--ligand_repr ./data/openfe/repr/torchdrug.lmdb \
--distance_threshold 20.0 \
--out_dir ./value/openfe \
--devices 4 \
--affinity_checkpoint "${CHECKPOINTS[@]}" \

sleep 10
pkill -f predict.py
sleep 10

torchrun --nproc_per_node=4 --rdzv_endpoint="localhost:29501" ./scripts/predict.py \
--data ./data/fep4/id.json \
--task value \
--structure ./data/fep4/pdb \
--structure_type pdb \
--ligand ./data/fep4/ligand_sdf.lmdb \
--ligand_type sdf \
--protein_repr ./data/fep4/repr/esm3.lmdb \
--ligand_repr ./data/fep4/repr/torchdrug.lmdb \
--distance_threshold 20.0 \
--out_dir ./value/fep4 \
--devices 4 \
--affinity_checkpoint "${CHECKPOINTS[@]}" \

sleep 10
pkill -f predict.py
sleep 10

torchrun --nproc_per_node=4 --rdzv_endpoint="localhost:29501" ./scripts/predict.py \
--data ./data/casp16/id.json \
--task value \
--structure ./data/casp16/pdb \
--structure_type pdb \
--ligand ./data/casp16/ligand_sdf.lmdb \
--ligand_type sdf \
--protein_repr ./data/casp16/repr/esm3.lmdb \
--ligand_repr ./data/casp16/repr/torchdrug.lmdb \
--distance_threshold 20.0 \
--out_dir ./value/casp16 \
--devices 4 \
--affinity_checkpoint "${CHECKPOINTS[@]}" \

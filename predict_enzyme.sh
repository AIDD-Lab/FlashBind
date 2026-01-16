#!/bin/bash

CHECKPOINTS_0=(
    "./checkpoints/enzyme_0_1.ckpt"
    "./checkpoints/enzyme_0_2.ckpt"
)

CHECKPOINTS_1=(
    "./checkpoints/enzyme_1_1.ckpt"
    "./checkpoints/enzyme_1_2.ckpt"
)

CHECKPOINTS_2=(
    "./checkpoints/enzyme_2_1.ckpt"
    "./checkpoints/enzyme_2_2.ckpt"
)

CHECKPOINTS_3=(
    "./checkpoints/enzyme_3_1.ckpt"
    "./checkpoints/enzyme_3_2.ckpt"
)

DATASET_NAME=(
    "Brenda"
    "DUF"
    "Esterase"
    "Glycosyltransferase"
    "Halogenase"
    "Nitrilase"
    "Phosphatase"
    "Thiolase"
)

for dataset in "${DATASET_NAME[@]}"; do

    torchrun --nproc_per_node=4 --rdzv_endpoint="localhost:29513" ./scripts/predict.py \
    --data ./data/ESIBank/${dataset}/data_split/test_id_0.json \
    --task enzyme \
    --structure ./data/ESIBank/${dataset}/protein_pdb.lmdb \
    --structure_type pdb \
    --ligand ./data/ESIBank/${dataset}/ligand_sdf.lmdb \
    --ligand_type sdf \
    --protein_repr ./data/ESIBank/${dataset}/repr/esm3.lmdb \
    --ligand_repr ./data/ESIBank/${dataset}/repr/torchdrug.lmdb \
    --distance_threshold 20.0 \
    --out_dir ./enzyme_predictions_0/${dataset} \
    --devices 4 \
    --affinity_checkpoint "${CHECKPOINTS_0[@]}" \
    --morgan_repr ./data/ESIBank/${dataset}/repr/morgan.lmdb \
    --unimol_repr ./data/ESIBank/${dataset}/repr/unimol.lmdb \

    sleep 5
    pkill -f predict.py
    sleep 5

done

for dataset in "${DATASET_NAME[@]}"; do

    torchrun --nproc_per_node=4 --rdzv_endpoint="localhost:29513" ./scripts/predict.py \
    --data ./data/ESIBank/${dataset}/data_split/test_id_1.json \
    --task enzyme \
    --structure ./data/ESIBank/${dataset}/protein_pdb.lmdb \
    --structure_type pdb \
    --ligand ./data/ESIBank/${dataset}/ligand_sdf.lmdb \
    --ligand_type sdf \
    --protein_repr ./data/ESIBank/${dataset}/repr/esm3.lmdb \
    --ligand_repr ./data/ESIBank/${dataset}/repr/torchdrug.lmdb \
    --distance_threshold 20.0 \
    --out_dir ./enzyme_predictions_1/${dataset} \
    --devices 4 \
    --affinity_checkpoint "${CHECKPOINTS_1[@]}" \
    --morgan_repr ./data/ESIBank/${dataset}/repr/morgan.lmdb \
    --unimol_repr ./data/ESIBank/${dataset}/repr/unimol.lmdb \

    sleep 5
    pkill -f predict.py
    sleep 5

done

for dataset in "${DATASET_NAME[@]}"; do

    torchrun --nproc_per_node=4 --rdzv_endpoint="localhost:29503" ./scripts/predict.py \
    --data ./data/ESIBank/${dataset}/data_split/test_id_2.json \
    --task enzyme \
    --structure ./data/ESIBank/${dataset}/protein_pdb.lmdb \
    --structure_type pdb \
    --ligand ./data/ESIBank/${dataset}/ligand_sdf.lmdb \
    --ligand_type sdf \
    --protein_repr ./data/ESIBank/${dataset}/repr/esm3.lmdb \
    --ligand_repr ./data/ESIBank/${dataset}/repr/torchdrug.lmdb \
    --distance_threshold 20.0 \
    --out_dir ./enzyme_predictions_2/${dataset} \
    --devices 4 \
    --affinity_checkpoint "${CHECKPOINTS_2[@]}" \
    --morgan_repr ./data/ESIBank/${dataset}/repr/morgan.lmdb \
    --unimol_repr ./data/ESIBank/${dataset}/repr/unimol.lmdb \

    sleep 5
    pkill -f predict.py
    sleep 5

done

for dataset in "${DATASET_NAME[@]}"; do

    torchrun --nproc_per_node=4 --rdzv_endpoint="localhost:29513" ./scripts/predict.py \
    --data ./data/ESIBank/${dataset}/data_split/test_id_3.json \
    --task enzyme \
    --structure ./data/ESIBank/${dataset}/protein_pdb.lmdb \
    --structure_type pdb \
    --ligand ./data/ESIBank/${dataset}/ligand_sdf.lmdb \
    --ligand_type sdf \
    --protein_repr ./data/ESIBank/${dataset}/repr/esm3.lmdb \
    --ligand_repr ./data/ESIBank/${dataset}/repr/torchdrug.lmdb \
    --distance_threshold 20.0 \
    --out_dir ./enzyme_predictions_3/${dataset} \
    --devices 4 \
    --affinity_checkpoint "${CHECKPOINTS_3[@]}" \
    --morgan_repr ./data/ESIBank/${dataset}/repr/morgan.lmdb \
    --unimol_repr ./data/ESIBank/${dataset}/repr/unimol.lmdb \

    sleep 5
    pkill -f predict.py
    sleep 5

done

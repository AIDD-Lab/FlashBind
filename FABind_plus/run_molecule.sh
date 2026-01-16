smiles_csv=./examples/smiles.csv
num_threads=0
save_pt_dir=./examples/repr_files

echo "======  preprocess molecules  ======"
python ./fabind/inference_preprocess_mol_confs.py --index_csv ${smiles_csv} --save_mols_dir ${save_pt_dir} --num_threads ${num_threads} --resume

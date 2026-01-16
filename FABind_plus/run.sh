work_dir=examples
index_csv=./${work_dir}/data.csv
save_pt_dir=./${work_dir}/repr_files
ckpt_path=./ckpt/fabind_plus_best_ckpt.bin
output_dir=./${work_dir}/inference_output

echo "======  inference begins  ======"
python ./fabind/inference_regression_fabind.py \
    --ckpt ${ckpt_path} \
    --batch_size 4 \
    --post-optim \
    --write-mol-to-file \
    --sdf-output-path-post-optim ${output_dir} \
    --index-csv ${index_csv} \
    --preprocess-dir ${save_pt_dir} \
    --instance-id 0
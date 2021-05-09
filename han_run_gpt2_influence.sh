#!/usr/bin/bash

# # influence function (gradient norm problematically huge? need scaling factor of 1e28)
# python gpt2_influence.py --output_dir=_test_influence_outputs --model_type=gpt2 --model_name_or_path=model/_test_output/ --task=contra\
#     --train_data_file=./data/contrastive_sample.tsv --train_data_field="N/A"\
#     --eval_data_file=./data/contrastive_sample.tsv --eval_data_field="N/A"\
#     --seed=2020 --per_gpu_train_batch_size=1\
#     --damping=1e-2 --scale=1e28 --lissa_repeat=1 --lissa_depth_pct=10.0\
#     --start_test_idx=0 --end_test_idx=2\
#     --logging_steps=10 --block_size=256\
#     --influence_metric="IF" --overwrite_output_dir --overwrite_cache

# # gradient cosine similarity
# python gpt2_influence.py --output_dir=_test_influence_outputs --model_type=gpt2 --model_name_or_path=model/_test_output/ --task=contra\
#     --train_data_file=./data/contrastive_sample.tsv --train_data_field="N/A"\
#     --eval_data_file=./data/contrastive_sample.tsv --eval_data_field="N/A"\
#     --seed=2020 --per_gpu_train_batch_size=1\
#     --start_test_idx=0 --end_test_idx=2\
#     --logging_steps=1 --block_size=256\
#     --influence_metric="GC" --overwrite_output_dir --overwrite_cache

########

# python gpt2_influence.py --output_dir=_test_influence_outputs --model_type=gpt2 --model_name_or_path=model/gpt2_ft_correct_output/ --task=contra\
#     --train_data_file=./data/train.txt --train_data_field="N/A"\
#     --eval_data_file=./data/test.txt --eval_data_field="N/A"\
#     --seed=2020 --per_gpu_train_batch_size=1\
#     --start_test_idx=0 --end_test_idx=9\
#     --logging_steps=1 --block_size=256\
#     --influence_metric="GC" --overwrite_output_dir --overwrite_cache
    
trap "kill 0" EXIT

# ngpu=$1 # first argument
ngpu=10 # first argument through hardcoding (for sbatch)

# for RTX 3090, processing 15 test examples in one run is possible
starti=("0" "10" "20" "30" "40" "50" "60" "70" "80" "90")
endi=("9" "19" "29" "39" "49" "59" "69" "79" "89" "99")

# regular tuning on various dataset setups
for i in $(seq 0 1 $((ngpu-1)))
do
CUDA_VISIBLE_DEVICES='0' python gpt2_influence.py --output_dir=GC_nocontra_outputs --model_type=gpt2 --model_name_or_path=model/gpt2_ft_correct_output/ --task=contra\
    --train_data_file=./data/train.txt --train_data_field="N/A"\
    --eval_data_file=./data/test.txt --eval_data_field="N/A"\
    --seed=2020 --per_gpu_train_batch_size=1\
    --start_test_idx=${starti[i]} --end_test_idx=${endi[i]}\
    --logging_steps=1 --block_size=256\
    --influence_metric="GC" --overwrite_output_dir --overwrite_cache --test_loss_no_contra

CUDA_VISIBLE_DEVICES='0' python gpt2_influence.py --output_dir=GC_outputs --model_type=gpt2 --model_name_or_path=model/gpt2_ft_correct_output/ --task=contra\
    --train_data_file=./data/train.txt --train_data_field="N/A"\
    --eval_data_file=./data/test.txt --eval_data_field="N/A"\
    --seed=2020 --per_gpu_train_batch_size=1\
    --start_test_idx=${starti[i]} --end_test_idx=${endi[i]}\
    --logging_steps=1 --block_size=256\
    --influence_metric="GC" --overwrite_output_dir --overwrite_cache
done

####

CUDA_VISIBLE_DEVICES='0' python gpt2_influence.py --output_dir=_test_influence_outputs --model_type=gpt2 --model_name_or_path=gpt2 --task=contra\
    --train_data_file=./data/train.txt --train_data_field="N/A"\
    --eval_data_file=./data/test.txt --eval_data_field="N/A"\
    --seed=2020 --per_gpu_train_batch_size=1\
    --start_test_idx=0 --end_test_idx=9\
    --logging_steps=1 --block_size=256\
    --influence_metric="GC" --overwrite_output_dir --overwrite_cache
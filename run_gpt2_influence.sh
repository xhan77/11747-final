python gpt2_influence.py --output_dir=_test_influence_outputs --model_type=gpt2 --model_name_or_path=model/_test_output/ --task=contra\
    --train_data_file=./data/contrastive_sample.tsv --train_data_field="N/A"\
    --eval_data_file=./data/contrastive_sample.tsv --eval_data_field="N/A"\
    --seed=2020 --per_gpu_train_batch_size=1\
    --damping=1e-2 --scale=1e28 --lissa_repeat=1 --lissa_depth_pct=10.0\
    --start_test_idx=0 --end_test_idx=2\
    --logging_steps=10 --block_size=256\
    --overwrite_output_dir --overwrite_cache

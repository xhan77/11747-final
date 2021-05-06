rm -rf gpt2_ft_correct_output/
rm -rf gpt2_ft_correct_tensorboard/

python gpt2_lm.py --output_dir=gpt2_ft_correct_output --model_type=gpt2 --model_name_or_path=gpt2 --task=contra\
    --do_train --train_data_file=../data/train.txt --train_data_field="N/A"\
    --do_eval --eval_data_file=../data/test.txt --eval_data_field="N/A"\
    --seed=2021 --num_train_epochs=20 --per_gpu_train_batch_size=6 --per_gpu_eval_batch_size=32\
    --learning_rate=5e-6 --weight_decay=1e-4 --warmup_steps=50 --gradient_accumulation_steps=32\
    --overwrite_output_dir --evaluate_during_training\
    --logging_steps=50 --block_size=256 --tensorboard_output_dir=gpt2_ft_correct_tensorboard/ #--overwrite_cache
    
# check eval trajectory using `tensorboard --logdir=gpt2_ft_correct_tensorboard`

# eval a trained model
python gpt2_lm.py --output_dir=gpt2_ft_correct_eval --model_type=gpt2 --model_name_or_path=gpt2_ft_correct_output/ --task=contra\
    --train_data_file=../data/train.txt --train_data_field="N/A"\
    --do_eval --eval_data_file=../data/test.txt --eval_data_field="N/A"\
    --seed=2021 --per_gpu_eval_batch_size=32 --overwrite_output_dir --evaluate_during_training\
    --logging_steps=50 --block_size=256 --tensorboard_output_dir=gpt2_ft_correct_tensorboard/ #--overwrite_cache

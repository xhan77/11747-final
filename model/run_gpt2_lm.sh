# below is a script for finetuning a GPT2 LM on MNLI premise data
rm -rf _test_tensorboard/
rm -rf _test_output/

python gpt2_lm.py --output_dir=_test_output --model_type=gpt2 --model_name_or_path=gpt2 --task=mnli\
    --do_train --train_data_file=../data/mnli_dev.tsv --train_data_field="premise"\
    --do_eval --eval_data_file=../data/mnli_dev.tsv --eval_data_field="premise"\
    --seed=2021 --num_train_epochs=20 --per_gpu_train_batch_size=6 --per_gpu_eval_batch_size=32\
    --learning_rate=5e-6 --weight_decay=1e-4 --warmup_steps=50 --gradient_accumulation_steps=32\
    --overwrite_output_dir --evaluate_during_training\
    --logging_steps=50 --block_size=64 --tensorboard_output_dir=_test_tensorboard/ --overwrite_cache

# check eval trajectory using `tensorboard --logdir=_test_tensorboard`


# eval with the contrastive setup
python gpt2_lm.py --output_dir=_test_output_2 --model_type=gpt2 --model_name_or_path=_test_output/ --task=contra\
    --train_data_file=../data/mnli_dev.tsv --train_data_field="premise"\
    --do_eval --eval_data_file=../data/contrastive_sample.tsv --eval_data_field="premise"\
    --seed=2021 --num_train_epochs=20 --per_gpu_train_batch_size=6 --per_gpu_eval_batch_size=1\
    --learning_rate=5e-6 --weight_decay=1e-4 --warmup_steps=50 --gradient_accumulation_steps=32\
    --overwrite_output_dir --evaluate_during_training\
    --logging_steps=50 --block_size=64 --tensorboard_output_dir=_test_tensorboard/ --overwrite_cache
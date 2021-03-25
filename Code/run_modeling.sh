#!/bin/bash
#to consider layer, we include one more argument and then change the output dir and corresponding command line arguments
#if [ "$#" -ne 4 ];    then
#    echo "Please provide name of the dataset, model name, configuration file and test file"
#    echo "first parameter is the dataset name (should be inside the folder logs/)"
#    echo "second parameter is the model name (should be bert/gpt)"
#    echo "third parameter is the configuration file (put the configuration file under logs/dataset/model name/config/)"
#    echo "fourth parameter is the test file/test script (evaluation will be based on this file)"
#    exit
#fi

root_dir=../Data/semanticscholar/
train_file="${root_dir}corpus.txt"
tokenizer="${root_dir}tokenizer"
output="${root_dir}model/"
config="${root_dir}config/"
model='bert --mlm'


# $3 is the configuration file. So we need to
config_file="${config}${3}"
conf=$(echo $3 | cut -d '.' -f1)
output="${output}${conf}"
test_file="${root_dir}${4}"
test_time_file="${root_dir}time_${4}"
#test_file="${train_file}"
export TRAIN_FILE=$train_file
export TEST_FILE=$test_file
#if [ -d $output ] && [ "$(ls -A $output)" ]; then
#    echo "Model directory is not empty. Assuming training is already done"
#else
    #echo "python3 run_mlm.py --output_dir=$output --model_type=$model --tokenizer_name=$tokenizer --learning_rate 1e-4 --do_train  --train_data_file $TRAIN_FILE  --gradient_accumulation_steps=4 --num_train_epochs 100 --per_gpu_train_batch_size 2 --save_steps 50000 --seed 42 --config_name=$config_file --line_by_line --do_eval --eval_data_file $TEST_FILE --block_size=8 --logging_steps 5000 --validation_split_percentage 0.2 --save_steps 50000  --save_total_limit 10 --mlm_probability 0.1"
    python run_mlm.py --output_dir=../Data/semanticscholar/model/ --model_type=bert --mlm_probability 0.1 --tokenizer_name=../Data/semanticscholar/tokenizer --learning_rate 1e-4 --do_train  --train_file ../Data/semanticscholar/corpus.txt  --gradient_accumulation_steps=4 --num_train_epochs 36 --per_gpu_train_batch_size 2 --save_steps 50000 --seed 42 --config_name=../Data/semanticscholar/config/ --line_by_line --do_eval --validation_file ../Data/semanticscholar/corpus.txt --max_seq_length=256 --logging_steps 5000 --save_steps 50000  --save_total_limit 10
    #echo "python3 run_language_modeling.py --output_dir=$output --model_type=$model --tokenizer_name=$tokenizer --learning_rate 1e-4 --do_train  --train_data_file $TRAIN_FILE  --gradient_accumulation_steps=4 --num_train_epochs 100 --per_gpu_train_batch_size 2 --save_steps 50000 --seed 42 --config_name=$config_file --line_by_line --do_eval --eval_data_file $TEST_FILE --train_time_file $train_time_file --test_time_file $test_time_file --block_size=16 --logging_steps 5000"
    #python3 run_language_modeling.py --output_dir=$output --model_type=$model --tokenizer_name=$tokenizer --learning_rate 1e-4 --do_train  --train_data_file $TRAIN_FILE  --gradient_accumulation_steps=4 --num_train_epochs 100 --per_gpu_train_batch_size 2 --save_steps 5000 --block_size=16 --seed 42 --config_name=$config_file --line_by_line --do_eval --eval_data_file $TEST_FILE --train_time_file $train_time_file --test_time_file $test_time_file
#fi

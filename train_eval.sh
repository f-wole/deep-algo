#!/bin/bash

train_start="[1980,1,15]" # format: (year, month)
train_end="[2002,10,20]"
valid_start="[2002,11,30]" # format: (year, month)
valid_end="[2010,3,1]"
test_start="[2016,3,14]" # format: (year, month)
test_end="[2020,7,5]" # format: (year, month)
window=10
profile="profiles/profile_mix_default"
batch=24

path="trial"
train_path=${path}"/data/train"
valid_path=${path}"/data/valid"
test_path=${path}"/data/test"
training_path=${path}"/training"
strategy_dir=${path}"/strategy"
evaluate_test=${path}"/evaluate/test"

rm -r ${path}
mkdir -p ${train_path}
mkdir -p ${valid_path}
mkdir -p ${test_path}
mkdir -p ${training_path}
mkdir -p ${strategy_dir}
mkdir -p ${evaluate_test}

python get_data.py ${train_start} ${train_end} ${mean} ${window} ${train_path} "true"
python get_data.py ${valid_start} ${valid_end} ${mean} ${window} ${valid_path} ${test_path} "false"
python get_data.py ${test_start} ${test_end} ${mean} ${window} ${test_path} "false"
python train.py ${train_path} ${valid_path} ${training_path} ${profile} ${batch} ${window}
python get_strategy.py ${valid_path} ${training_path} ${profile} ${strategy_dir}
python evaluate.py ${training_path} ${test_path} ${evaluate_test} ${profile} ${strategy_dir}
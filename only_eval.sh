#!/bin/bash

train_start="[1980,1]" # format: (year, month)
train_end="[2002,10]"
valid_start="[2002,11]" # format: (year, month)
valid_end="[2010,3]"
test_start="[2016,3]" # format: (year, month)
test_end="[2020,7]" # format: (year, month)
mean="true"
window=8
profile="profiles/profile_lstm_default"
batch=24
path="trial"
train_path=${path}"/data/train"
valid_path=${path}"/data/valid"
test_path=${path}"/data/test"
training_path=${path}"/training"
strategy_dir=${path}"/strategy"
evaluate_test=${path}"/evaluate/test"

rm ${strategy_dir}
rm ${evaluate_test}
mkdir -p ${strategy_dir}
mkdir -p ${evaluate_test}

python get_strategy.py ${valid_path} ${training_path} ${profile} ${strategy_dir}
python evaluate.py ${training_path} ${test_path} ${evaluate_test} ${profile} ${strategy_dir}
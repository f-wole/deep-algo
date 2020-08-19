#!/bin/bash

train_start="[1980,1]" # format: (year, month)
train_end="[2002,10]"
valid_start="[2002,11]" # format: (year, month)
valid_end="[2010,3]"
test_start="[2016,3]" # format: (year, month)
test_end="[2020,7]" # format: (year, month)
mean="true"
window=5
train_path="trial/data/train.pkl"
valid_path="trial/data/valid.pkl"
test_path="trial/data/test.pkl"
out_dir="trial/trainining"
profile="profiles/profile_lstm_default"
batch=24
strategy_dir="trial/strategy"
evaluate_dir="trial/evaluate"

python get_data.py ${train_start} ${train_end} ${mean} ${window} ${train_path} "true"
python get_data.py ${valid_start} ${valid_end} ${mean} ${window} ${valid_path} ${test_path} "false"
python get_data.py ${test_start} ${test_end} ${mean} ${window} ${test_path} "false"
python train.py ${train_path} ${valid_path} ${out_dir} ${profile} ${batch}
python get_strategy.py ${valid_path} ${out_dir} ${profile} ${strategy_dir}
python evaluate.py ${out_dir} ${test_path} ${evaluate_dir} ${profile} ${strategy_dir}
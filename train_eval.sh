#!/bin/bash

train_start="[1980,1]" # format: (year, month)
valid_start="[2000,2]" # format: (year, month)
test_start="[2010,3]" # format: (year, month)
test_end="[2018,6]" # format: (year, month)
mean="true"
window=5
train_path="prove/train.pkl"
valid_path="prove/valid.pkl"
test_path="prove/test.pkl"
out_dir="prove/model"
profile="profiles/profile_lstm_default"
batch=24

python get_data.py ${train_start} ${valid_start} ${test_start} ${test_end} ${mean} ${window} ${train_path} ${valid_path} ${test_path}
#python train.py ${train_path} ${valid_path} ${out_dir} ${profile} ${batch}
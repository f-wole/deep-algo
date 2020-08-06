#!/bin/bash

start="[1973,1,1]"
end="[2020,4,3]"
ttratio=0.8
out_dir="first_try"
model_type="lstm"
profile="profiles/profile_lstm_default"
batch=24
data_api="yahoo"

python train.py ${start} ${end} ${ttratio} ${out_dir} ${model_type} ${profile} ${batch} ${data_api}
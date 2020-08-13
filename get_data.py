import os
import sys
import json
import pickle
from utils import get_data_yahoo

train_start=json.loads(sys.argv[1]) # format: (year, month) included
valid_start=json.loads(sys.argv[2]) # format: (year, month) included
test_start=json.loads(sys.argv[3]) # format: (year, month) included
test_end=json.loads(sys.argv[4]) # format: (year, month) excluded
mean=sys.argv[5] in ["true","True"]
window=int(sys.argv[6])
train_path=sys.argv[7]
valid_path=sys.argv[8]
test_path=sys.argv[9]

### get data and process it
# salvo un pickle file con: window, pandas dataframes e versioni tensorizzate per NN

# FIX DATES
last_days={1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
dfm=get_data_yahoo([train_start[0],train_start[1],1],
                   [test_end[0],test_end[1],last_days[test_end[1]]],
                   window)

with open(train_path,"wb") as w:
    pickle.dump(dfm,w)
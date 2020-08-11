import os
import sys
import pickle

train_start=sys.argv[1]
valid_start=sys.argv[2]
test_start=sys.argv[3]
test_end=sys.argv[4]
mean=sys.argv[5]
window=sys.argv[6]
train_path=sys.argv[7]
valid_path=sys.argv[8]
test_path=sys.argv[9]

### get data and process it
# qual'è la cosa migliore, salvare tutto dopo aver tensorizzato e fatto min-max oppure salvare direttamente
# i pandas dataframes e processare in ogni script? 
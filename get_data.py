import os
import sys
import json
import pickle
from utils import get_data_yahoo, process_data
import matplotlib.pyplot as plt

start=json.loads(sys.argv[1]) # format: (year, month) included
end=json.loads(sys.argv[2]) # format: (year, month) excluded
mean=sys.argv[3] in ["true","True"]
window=int(sys.argv[4])
save_path=sys.argv[5]
train=sys.argv[6] in ["true","True"]

### get data and process it
# salvo un pickle file con: window, pandas dataframes e versioni tensorizzate per NN

if not save_path.endswith("/"):
    save_path+="/"

# fix start
start[0]-=2
if train:
    start[1]-=window
else:
    start[1]-=window+1
if start[1]<=0:
    start[1]+=12
    start[0]-=1

# FIX DATES
last_days={1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
dfm=get_data_yahoo([start[0],start[1],1],
                   [end[0],end[1],last_days[end[1]]],
                   window)

X,y=process_data(dfm,window)
dfm=dfm[window:]


with open(save_path+"data.pkl","wb") as w:
    pickle.dump([X,y,dfm,window],w)

dfm.to_excel(save_path+"data.xlsx")
dfu=dfm.index
dfu.drop_duplicates
dfu=list(dfu)

with open(save_path+"dates","w",encoding="utf8") as w:
    for date in dfu:
        date=str(date)
        date="-".join(date.split("-")[:2])
        w.write(date+"\n")

print("### Successfully saved to ",save_path)
print("\t X shape = ",X.shape)
print("\t y shape = ",y.shape)
print("\t dfm shape = ",dfm.shape)

plt.plot(dfm["fd_cm_close"])
plt.title("Close price of first day of current month")
plt.xticks(rotation=45)
plt.savefig(save_path+"plot")
import os
import sys
import json
import pickle
from utils import get_data_yahoo, process_data
import matplotlib.pyplot as plt

start=json.loads(sys.argv[1]) # format: (year, month) included
end=json.loads(sys.argv[2]) # format: (year, month) excluded
window=int(sys.argv[3])
save_path=sys.argv[4]
train=sys.argv[5] in ["true","True"]

### get data and process it
# salvo un pickle file con: window, pandas dataframes e versioni tensorizzate per NN

if not save_path.endswith("/"):
    save_path+="/"

dfm=get_data_yahoo([start[0],start[1],start[2]],
                   [end[0],end[1],end[2]],
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

plt.plot(dfm["Open"])
plt.title("Close price of first day of current month")
plt.xticks(rotation=45)
plt.savefig(save_path+"plot")
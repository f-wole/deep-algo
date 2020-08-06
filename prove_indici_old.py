import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr

# start=[1980,1,1]
start=[2010,5,28]
end=[2020,5,1]
start_date = datetime.datetime(start[0],start[1],start[2])
end_date = datetime.datetime(end[0],end[1],end[2])
# index="^GSPC" # S&P500 index
# index="SWDA.MI" # MSCI World Milano
# index="IWDA.AS" # MSCI World Amsterdam
index="SXR8.DE" # S&P500 ETF German in €
# index="IUSE.MI" # S&P500 ETF Milano in €

# df = pdr.get_data_yahoo(index, start=start_date, end=end_date)
# print(df)

ind = pdr.get_data_yahoo("^GSPC", start=start_date, end=end_date)
etf = pdr.get_data_yahoo("IUSE.MI", start=start_date, end=end_date)
# print(etf)
etf_c=etf["Close"]
ind_c=[]
for i,date in enumerate(list(etf.index)):
    ind_c.append(list(ind.loc[ind.index==date,"Close"]))
etf_cc,ind_cc=[],[]
for i,x in enumerate(ind_c):
    if x!=[]:
        etf_cc.append(etf_c[i])
        ind_cc.append(x)
ind_cc=np.array(ind_cc)
etf_cc=np.array(etf_cc)
print(ind_cc[0])
print(len(etf_cc),len(ind_cc))

etf_cc/=etf_cc[0]
ind_cc/=ind_cc[0]

plt.plot(etf_cc)
plt.plot(ind_cc)
plt.show()
from utils import get_data_yahoo,process_data

start=[2010,5,28]
end=[2012,5,1]
window=10

df=get_data_yahoo(False,start,end,10,"^GSPC")

df.to_excel("df.xlsx")

X,y=process_data(df,10,"Close")
print(X.shape)
print(y.shape)
print(X)
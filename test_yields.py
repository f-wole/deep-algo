import numpy as np
import pandas as pd
from utils import yield_gross

df={}
df["quot"]=np.array([0.9,1.1,0.8,0.5,1.2,1.3,1.4])
df=pd.DataFrame(df)
v=np.array([1,0,1,1,0,0,1])

assert df["quot"].shape[0]==v.shape[0]

print(yield_gross(df,v))
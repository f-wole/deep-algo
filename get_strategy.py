import numpy as np
import pickle
import sys
import keras
from utils import yield_net
from scipy import optimize

valid_path=sys.argv[1]
out_dir=sys.argv[2]
profile=sys.argv[3]

if not out_dir.endswith("/"):
    out_dir=out_dir+"/"
model_type=profile.split("_")[1]
model_path=out_dir+"model_"+model_type+".h5"

model=keras.models.load_model(model_path)
with open(valid_path,"rb") as r:
    X,y,dfm,window=pickle.load(r)

dfm=dfm[1:]

y_pred=model.predict(X)
y_pred_resh=y_pred.reshape(y_pred.shape[0])


def to_optimize(z):
    a,b=z
    w=y_pred_resh[1:]-(a*y_pred_resh[:-1]+b)
    v=np.maximum(np.sign(w), 0)
    return -yield_net(dfm,v)[0]

rranges = (slice(-1, 2, 0.02), slice(-1, 2, 0.02))
resbrute = optimize.brute(to_optimize, rranges, full_output=True, finish=optimize.fmin)
print("best parameters are : ", resbrute[0])
a,b=resbrute[0]
w=y_pred_resh[1:]-(a*y_pred_resh[:-1]+b)
v=np.maximum(np.sign(w), 0)
print("total percentage gain : ",yield_net(dfm,v)[0])
print("Annual percentage gain : ",yield_net(dfm,v)[1])
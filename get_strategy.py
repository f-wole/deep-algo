import numpy as np
import pickle
import os
import sys
import keras
from utils import yield_net,get_ins,get_outs
from scipy import optimize
import matplotlib.pyplot as plt

valid_path=sys.argv[1]
model_dir=sys.argv[2]
profile=sys.argv[3]
out_dir=sys.argv[4]

if not out_dir.endswith("/"):
    out_dir=out_dir+"/"
if not model_dir.endswith("/"):
    model_dir=model_dir+"/"
if not valid_path.endswith("/"):
    valid_path+="/"
model_type=profile.split("_")[1]
model_path=model_dir+"model_"+model_type+".h5"

model=keras.models.load_model(model_path)
with open(valid_path+"data.pkl","rb") as r:
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
print("total percentage gain on validation: ",yield_net(dfm,v)[0])
print("Annual percentage gain on validation: ",yield_net(dfm,v)[1])

w_sim=np.diff(y_pred.reshape(y_pred.shape[0]),1)
v_sim=np.maximum(np.sign(w_sim),0)

with open(out_dir+"parameters","w",encoding="utf8") as w:
    w.write("a = "+str(a)+"\n")
    w.write("b = "+str(b)+"\n")
    w.write("total percentage gain on validation: "+str(yield_net(dfm, v)[0])+"\n")
    w.write("Annual percentage gain on validation: "+str(yield_net(dfm, v)[1])+"\n\n")
    w.write("With a = -1 and b = 0:\n")
    w.write("total percentage gain on validation: " + str(yield_net(dfm, v_sim)[0]) + "\n")
    w.write("Annual percentage gain on validation: " + str(yield_net(dfm, v_sim)[1]) + "\n\n")


assert(y[1:].shape[0]==v.shape[0])
assert(y_pred[1:].shape[0]==y[1:].shape[0])

plt.figure(figsize=(30,10))
plt.plot(y[1:], label="actual")
plt.plot(y_pred[1:], label="prediction lstm")
plt.plot(get_ins(y[1:],v)[0],get_ins(y[1:],v)[1],'^', markersize=10, color='g',label="Tuned in")
plt.plot(get_outs(y[1:],v)[0],get_outs(y[1:],v)[1],'v', markersize=10, color='r',label="Tuned out")
plt.plot(get_ins(y[1:],v_sim)[0],get_ins(y[1:],v_sim)[1],'^', markersize=10, color='b',label="Simple in")
plt.plot(get_outs(y[1:],v_sim)[0],get_outs(y[1:],v_sim)[1],'v', markersize=10, color='y',label="Simple out")
plt.legend(fontsize=20)
plt.grid(axis="both")
plt.title("Validation",fontsize=25)
plt.savefig(out_dir+"tuning.png")

with open(out_dir+"params.pkl","wb") as w:
    pickle.dump([a,b],w)
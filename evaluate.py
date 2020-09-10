from keras.models import load_model
import sys
import os
import pickle
from utils import yield_net,get_ins,get_outs
import numpy as np
import matplotlib.pyplot as plt

model_dir=sys.argv[1]
test_path=sys.argv[2]
out_dir=sys.argv[3]
profile=sys.argv[4]
strategy_dir=sys.argv[5]

if not strategy_dir.endswith("/"):
    strategy_dir=strategy_dir+"/"
if not out_dir.endswith("/"):
    out_dir=out_dir+"/"
if not model_dir.endswith("/"):
    model_dir=model_dir+"/"
if not test_path.endswith("/"):
    test_path+="/"

model_type=profile.split("_")[1]
model_path=model_dir+"model_"+model_type+".h5"
model=load_model(model_path)

with open(strategy_dir+"params.pkl","rb") as r:
    a,b=pickle.load(r)

with open(test_path+"data.pkl","rb") as r:
    X,y,dfm,window=pickle.load(r)
dfm=dfm[1:]

y_pred=model.predict(X)
y_pred_resh=y_pred.reshape(y_pred.shape[0])


w=y_pred_resh[1:]-(a*y_pred_resh[:-1]+b)
v=np.maximum(np.sign(w), 0)
print("total percentage gain on test: ",yield_net(dfm,v)[0])
print("Annual percentage gain on test:: ",yield_net(dfm,v)[1])

w_sim=np.diff(y_pred.reshape(y_pred.shape[0]),1)
v_sim=np.maximum(np.sign(w_sim),0)
v_buyhold=np.ones(v_sim.shape[0])

with open(out_dir+"parameters","w",encoding="utf8") as w:
    w.write("a = "+str(a)+"\n")
    w.write("b = "+str(b)+"\n")
    w.write("total percentage gain on test: "+str(yield_net(dfm, v)[0])+"\n")
    w.write("Annual percentage gain on test: "+str(yield_net(dfm, v)[1])+"\n\n")
    w.write("With a = -1 and b = 0:\n")
    w.write("total percentage gain on test: " + str(yield_net(dfm, v_sim)[0]) + "\n")
    w.write("Annual percentage gain on test: " + str(yield_net(dfm, v_sim)[1]) + "\n\n")
    # w.write("Kiss:\n")
    # w.write("total percentage gain on test: " + str(yield_net(dfm, v_ma)[0]) + "\n")
    # w.write("Annual percentage gain on test: " + str(yield_net(dfm, v_ma)[1]) + "\n\n")
    w.write("Buy and hold:\n")
    w.write("total percentage gain on test: " + str(yield_net(dfm, v_buyhold)[0]) + "\n")
    w.write("Annual percentage gain on test: " + str(yield_net(dfm, v_buyhold)[1]) + "\n\n")


assert(y[1:].shape[0]==v.shape[0])
assert(y_pred[1:].shape[0]==y[1:].shape[0])

plt.figure(figsize=(30,10))
plt.plot(y[1:], label="actual")
plt.plot(y_pred[1:], label="prediction lstm")
plt.plot(get_ins(y[1:],v)[0],get_ins(y[1:],v)[1],'^', markersize=10, color='g',label="Tuned in")
plt.plot(get_outs(y[1:],v)[0],get_outs(y[1:],v)[1],'v', markersize=10, color='r',label="Tuned out")
# plt.plot(get_ins(y[1:],v_sim)[0],get_ins(y[1:],v_sim)[1],'^', markersize=10, color='b',label="Simple in")
# plt.plot(get_outs(y[1:],v_sim)[0],get_outs(y[1:],v_sim)[1],'v', markersize=10, color='y',label="Simple out")

plt.legend(fontsize=20)
plt.grid(axis="both")
plt.title("Test",fontsize=25)
plt.savefig(out_dir+"tuning.png")


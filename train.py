import os
import sys
import pickle
import configparser
from ast import literal_eval
from utils import get_data_yahoo,model_lstm,model_mix,process_data,process_data_test
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard,ReduceLROnPlateau
import matplotlib.pyplot as plt

start=literal_eval(sys.argv[1]) # format is year, month, day
end=literal_eval(sys.argv[2])
ttratio=float(sys.argv[3])
out_dir=sys.argv[4]
model_type=sys.argv[5]
profile=sys.argv[6]
batch=int(sys.argv[7])
data_api=sys.argv[8]

if not out_dir.endswith("/"):
    out_dir=out_dir+"/"
os.system("rm -r "+out_dir)
os.system("mkdir "+out_dir)

# Get data, divide in train & test and save it

if data_api=="yahoo":
    df,dfm=get_data_yahoo(start,end,'^GSPC')
num=int(dfm.shape[0])
df_train=dfm.iloc[:int(ttratio*num),:]
df_test=dfm.iloc[int(ttratio*num):,:]
df_train.to_excel(out_dir+"train.xlsx")
df_test.to_excel(out_dir+"test.xlsx")
with open(out_dir+"data.pic","wb") as w:
    pickle.dump([df_train,df_test,model_type,profile,data_api],w)
if data_api=="yahoo":
    features=8

# Inizialize model

if model_type not in ["lstm","mix"]:
    print("Error: model_type can be either 'lstm' or 'mix'")
    exit(1)
config = configparser.RawConfigParser()
config.read(profile)
if model_type=="lstm":
    lstm1 = int(config.get('NetworkProperties', 'lstm1'))
    lstm2 = int(config.get('NetworkProperties', 'lstm2'))
    dense = int(config.get('NetworkProperties', 'dense'))
    drop_out = float(config.get('NetworkProperties', 'drop_out'))
    n_epochs = int(config.get('NetworkProperties', 'n_epochs'))
    lr = float(config.get('NetworkProperties', 'lr'))
    rop = config.get('NetworkProperties', 'rop') in ["true","True"]
    window = int(config.get('NetworkProperties', 'window'))

    model=model_lstm(window, features,lstm1,lstm2,dense,drop_out,lr)

if model_type=="mix":
    filters=int(config.get('NetworkProperties', 'filters'))
    ksize=int(config.get('NetworkProperties', 'ksize'))
    lstm1 = int(config.get('NetworkProperties', 'lstm1'))
    lstm2 = int(config.get('NetworkProperties', 'lstm2'))
    dense = int(config.get('NetworkProperties', 'dense'))
    drop_out = float(config.get('NetworkProperties', 'drop_out'))
    n_epochs = int(config.get('NetworkProperties', 'n_epochs'))
    lr = float(config.get('NetworkProperties', 'lr'))
    rop = config.get('NetworkProperties', 'rop') in ["true","True"]
    window = int(config.get('NetworkProperties', 'window'))

    model=model_mix(window, features,filters,ksize,lstm1,lstm2,dense,drop_out,lr)

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=25, verbose=1,factor=0.25, min_lr=0.00001)
callbacks=[]
if rop:
    callbacks.append(learning_rate_reduction)


# Preprocess data to feed it to the model
X_train,y_train=process_data_test(df_train,2,data_api)
X_train,y_train=process_data(df_train,window,data_api)
X_test,y_test=process_data(df_test,window,data_api)
print("# X_Train: ",X_train.shape)
print("# y_Train: ",y_train.shape)
print("# X_Test: ",X_test.shape)
print("# y_Test: ",y_test.shape)

# Train

history_lstm=model.fit(X_train,y_train,epochs=n_epochs, batch_size=batch, validation_data=(X_test, y_test),
                       verbose=1, callbacks=callbacks,shuffle=False)

plt.plot(history_lstm.history['loss'])
plt.plot(history_lstm.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig(out_dir+"train_plot.png")

model.save_weights(out_dir+"model_"+model_type+".h5")
import numpy as np
import datetime
import pickle
import pandas as pd
import pandas_datareader as pdr
from keras.models import Sequential
from keras.optimizers import RMSprop,Adam
from keras.layers import Dense,Dropout,BatchNormalization,Conv1D,Flatten,MaxPooling1D,LSTM
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def get_data_yahoo(start,end,window,mean=True,index='^GSPC'):

    # input format: (year, day, month)

    start_date = datetime.datetime(start[0], start[1], start[2])
    end_date = datetime.datetime(end[0], end[1], end[2])
    df = pdr.get_data_yahoo(index, start=start_date, end=end_date)
    to_drop = ["High", "Low", "Adj Close"]
    df.drop(to_drop, axis=1, inplace=True)

    df["weekday"] = df.index.weekday  # The day of the week with Monday=0, Sunday=6.

    df["Close_30"] = df["Close"].rolling(window=30).mean().shift(1)
    df["Close_150"] = df["Close"].rolling(window=150).mean().shift(1)
    df["Open_30"] = df["Open"].rolling(window=30).mean().shift(1)
    df["Open_150"] = df["Open"].rolling(window=150).mean().shift(1)
    df["Volume_30"] = df["Volume"].rolling(window=30).mean().shift(1)
    df["Volume_150"] = df["Volume"].rolling(window=150).mean().shift(1)

    df = df[150:]
    df = df.loc[df["weekday"] == 0]
    df["Open next"] = df["Open"].shift(-1)
    df["quot"] = df["Open next"] / df["Open"]
    df = df[:-1]

    return df


def model_lstm(window, features,lstm1,lstm2,dense,drop_out,lr):
    model = Sequential()
    model.add(LSTM(lstm1, input_shape=(window+1, features), return_sequences=True))
    model.add(Dropout(drop_out))
    model.add(LSTM(lstm2, return_sequences=False))  # there is no need to specify input_shape here
    model.add(Dropout(drop_out))
    model.add(Dense(dense, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    model.summary()

    return model


def model_mix(window, features,filters,ksize,lstm1,lstm2,dense,drop_out,lr):
    model = Sequential()
    model.add(
        Conv1D(input_shape=(window+1, features), filters=filters, kernel_size=ksize, strides=1, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same'))
    model.add(LSTM(lstm1, return_sequences=True))
    model.add(Dropout(drop_out))
    model.add(LSTM(lstm2, return_sequences=False))
    model.add(Dropout(drop_out))
    model.add(Dense(dense, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    model.summary()

    return model


def process_data(df,window):
    def create_window(data, window_size=1):
        data_s = data.copy()
        for i in range(window_size):
            data = pd.concat([data, data_s.shift(-(i + 1))], axis=1)

        data.dropna(axis=0, inplace=True)
        return (data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    lista=["Open","Open_30","Open_150","Close","Close_30","Close_150","Volume","Volume_30",
           "Volume_150","Open next"]
    dg = pd.DataFrame(scaler.fit_transform(df[lista].values))
    X = dg[[0, 1, 2, 3, 4, 5, 6, 7,8]]
    X = create_window(X, window)
    X = np.reshape(X.values, (X.shape[0], window + 1, 9))

    y = np.array(dg[8][window:])

    return X, y



def yield_gross(df,v):
    ## df["quot"] è il rapporto tra il prezzo open del primo giorno del mese successivo con
    ## il prezzo open del primo giorno del mese corrente
    ## v è il vettore che indica se sei o no nel mercato in quel mese
    prod=(v*df["quot"]+1-v).prod()
    n_years=len(v)/(12*4)
    return (prod-1)*100,((prod**(1/n_years))-1)*100


def separate_ones(u):
    u_ = np.r_[0, u, 0]
    i = np.flatnonzero(u_[:-1] != u_[1:])
    v, w = i[::2], i[1::2]
    if len(v) == 0:
        return np.zeros(len(u)), 0

    n, m = len(v), len(u)
    o = np.zeros(n * m, dtype=int)

    r = np.arange(n) * m
    o[v + r] = 1

    if w[-1] == m:
        o[w[:-1] + r[:-1]] = -1
    else:
        o[w + r] -= 1

    out = o.cumsum().reshape(n, -1)
    return out, n


def yield_net(df, v,tax_cg=0.26,comm_bk=0.001):
    n_years = len(v) / (12*4)

    w, n = separate_ones(v)
    A = (w * np.array(df["quot"]) + (1 - w)).prod(axis=-1)  # A is the product of each group of ones of 1 for df["quot"]
    A1p = np.maximum(0, np.sign(A - 1))  # vector of ones where the corresponding element if  A  is > 1, other are 0
    Ap = A * A1p  # vector of elements of A > 1, other are 0
    Am = A - Ap  # vector of elements of A <= 1, other are 0
    An = Am + (Ap - A1p) * (1 - tax_cg) + A1p
    prod = An.prod() * ((1 - comm_bk) ** (2 * n))

    return (prod - 1) * 100, ((prod ** (1 / n_years)) - 1) * 100


def get_ins(npdata,v):
    x,y=[],[]
    for i in range(npdata.shape[0]):
        if v[i]==1:
            if i==0:
                x.append(i)
                y.append(npdata[i])
            else:
                if v[i-1]==0:
                    x.append(i)
                    y.append(npdata[i])
    # df=pd.DataFrame({"index":pd.DatetimeIndex(x),"value":np.array(y)})
    # df.index=pd.to_datetime(df.index)
    # df.set_index('index',inplace=True)
    return x,y

def get_outs(npdata,v):
    x, y = [], []
    for i in range(npdata.shape[0]):
        if v[i] == 0:
            if i == 0:
                x.append(i)
                y.append(npdata[i])
            else:
                if v[i - 1] == 1:
                    x.append(i)
                    y.append(npdata[i])
    # df=pd.DataFrame({"index":pd.DatetimeIndex(x),"value":np.array(y)})
    # df.index=pd.to_datetime(df.index)
    # df.set_index('index',inplace=True)
    return x, y
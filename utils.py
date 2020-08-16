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

    #
    # returns indices and moving averages for all months between start (included) and end (excluded)
    # Tested

    # if mean:
    #     # per ora non considero window
    #     start_date = datetime.datetime(start[0], start[1], start[2])
    #     great_start_date = datetime.datetime(start[0]-2, start[1], start[2])
    #     end_date = datetime.datetime(end[0], end[1], end[2])
    #     print("start ",start_date)
    #     print("end ",end_date)
    #     df = pdr.get_data_yahoo(index, start=great_start_date, end=end_date)
    #     lista_date=list(df.index)
    #     str_ind=[i for i,date in enumerate(lista_date) if date==start_date][0]
    #     window_start=lista_date[str_ind-window]
    #     print("window_start ",window_start)
    #     df = pdr.get_data_yahoo(index, start=window_start, end=end_date)
    #     return df
    start_date = datetime.datetime(start[0],start[1],start[2])
    end_date = datetime.datetime(end[0],end[1],end[2])
    df = pdr.get_data_yahoo(index, start=start_date, end=end_date)
    df.drop("Adj Close", axis=1, inplace=True)

    first_days = []
    # First year
    for month in range(start[1], 13):
        first_days.append(min(df[str(start[0]) + "-" + str(month)].index))
    # Other years
    for year in range(start[0] + 1, end[0]):
        for month in range(1, 13):
            first_days.append(min(df[str(year) + "-" + str(month)].index))
    # Last year
    for month in range(1, end[1] + 1):
        first_days.append(min(df[str(end[0]) + "-" + str(month)].index))

    dfm = df.resample("M").mean()
    dfm = dfm[:-1]  # As we said, we do not consider the month of end_date

    dfm["fd_cm"] = first_days[:-1]
    dfm["fd_nm"] = first_days[1:]
    dfm["fd_cm_open"] = np.array(df.loc[first_days[:-1], "Open"])
    dfm["fd_nm_open"] = np.array(df.loc[first_days[1:], "Open"])
    dfm["ratio"] = dfm["fd_nm_open"].divide(dfm["fd_cm_open"])

    dfm["mv_avg_12"] = dfm["Open"].rolling(window=12).mean().shift(1)
    dfm["mv_avg_24"] = dfm["Open"].rolling(window=24).mean().shift(1)
    dfm["quot"] = dfm["fd_nm_open"].divide(dfm["fd_cm_open"])

    dfm = dfm.iloc[24:, :]  # we remove the first 24 months, since they do not have the 2-year moving average

    indexes=["High","Low","Open","Close","Volume"]
    for index in indexes:
        dfm[index+"_avg"]=dfm[index]
        dfm=dfm.drop(index,axis=1)

    return dfm


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
    dg = pd.DataFrame(scaler.fit_transform(df[["High_avg", "Low_avg", "Open_avg", "Close_avg", "Volume_avg", "fd_cm_open",
                                               "mv_avg_12", "mv_avg_24", "fd_nm_open"]].values))
    X = dg[[0, 1, 2, 3, 4, 5, 6, 7]]
    X = create_window(X, window)
    X = np.reshape(X.values, (X.shape[0], window + 1, 8))

    y = np.array(dg[8][window:])

    return X, y


def process_data_mod(df,window,y_label):
    def create_window(data, window_size=1):
        data_s = data.copy()
        for i in range(window_size):
            data = pd.concat([data, data_s.shift(-(i + 1))], axis=1)

        data.dropna(axis=0, inplace=True)
        return (data)

    cols=len(list(df.columns))
    ind_y=[i for i,lb in enumerate(df.columns) if lb==y_label][0]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled=pd.DataFrame(scaler.fit_transform(df.values))
    not_scaled=pd.DataFrame(df.values)
    dg = pd.DataFrame(scaler.fit_transform(df[["High_avg", "Low_avg", "Open_avg", "Close_avg",
                                                    "Volume_avg", "fd_cm_open",
                                                    "mv_avg_12", "mv_avg_24", "fd_nm_open"]].values))


    X = create_window(scaled, window)
    X = np.reshape(X.values, (X.shape[0], window + 1, cols))

    y = np.array(not_scaled[ind_y][window:])

    return X,y

def yield_gross(df,v):
    ## df["quot"] è il rapporto tra il prezzo open del primo giorno del mese successivo con
    ## il prezzo open del primo giorno del mese corrente
    ## v è il vettore che indica se sei o no nel mercato in quel mese
    prod=(v*df["quot"]+1-v).prod()
    n_years=len(v)/12
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


def yield_net(df, v,tax_cg=0.26,comm_bk=0.01):
    n_years = len(v) / 12

    w, n = separate_ones(v)
    A = (w * np.array(df["quot"]) + (1 - w)).prod(axis=-1)  # A is the product of each group of ones of 1 for df["quot"]
    A1p = np.maximum(0, np.sign(A - 1))  # vector of ones where the corresponding element if  A  is > 1, other are 0
    Ap = A * A1p  # vector of elements of A > 1, other are 0
    Am = A - Ap  # vector of elements of A <= 1, other are 0
    An = Am + (Ap - A1p) * (1 - tax_cg) + A1p
    prod = An.prod() * ((1 - comm_bk) ** (2 * n))

    return (prod - 1) * 100, ((prod ** (1 / n_years)) - 1) * 100
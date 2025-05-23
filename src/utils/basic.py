import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import random
import math
from operator import itemgetter
import statsmodels

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model# type: ignore #, Input
from keras.layers import Input # type: ignore
from keras.constraints import max_norm, unit_norm # type: ignore
from keras.layers import Dense, Flatten, SpatialDropout1D, Activation, Add, BatchNormalization, Conv1D, MaxPooling1D # type: ignore
from keras import regularizers
from keras.optimizers import Adam, SGD # type: ignore
from keras.callbacks import EarlyStopping # type: ignore
from tcn import TCN
# from tcn import compiled_tcn


def random_CNN1():
    lags = random.randint(1, 50)   # quantidade de lags
    if lags <= 2:
        max_pool = 2
    else:
        max_pool = min(5,lags)
    return genotypeCNN1(
    random.randint(0, 2),   # número de filtros [16,32,64]
    random.randint(0, 2),   # probabilidade de pooling [0%, 50%, 100%]
    random.uniform(0, 0.5),  # porcentagem dropout
    random.randint(0, 1),   # normalização (0 - sim, 1 - não)
    lags,   # quantidade de lags
    random.randint(1, 5), # número de camadas convolucionais
    random.randint(0, 3), # tamanho do kernel de convolução
    random.randint(2, max_pool), #tamanho da janela de pooling [2,3,4,5] 
    0 #Tipo
    )

def random_CNN2():
    """
    Cria genótipo aleatório CNN2
    :return: o genótipo, um dicionário com todos os hiperparâmetros
    """
    num_conv = random.randint(1, 5)
    k = random.randint(0, 3)
  
    if k == 0: 
        kernel_size = 2
    elif k == 1:
        kernel_size = 3
    elif k == 2:
        kernel_size = 5
    else:
        kernel_size = 11
    
    max_lags = (2**num_conv)*kernel_size
    min_lags = (2**num_conv)
  
    return genotypeCNN2(
    random.randint(0, 2), 
    random.uniform(0, 0.5),
    random.randint(0, 1),
    random.randint(min_lags, max_lags), 
    num_conv,
    kernel_size,
    1
    )
    
def random_CNN3():
    """
    Cria genótipo aleatório CNN3
    :return: o genótipo, um dicionário com todos os hiperparâmetros
    """
    num_conv = random.randint(1, 5) 
    k = random.randint(0, 3) 
    pilhas = random.randint(1, 2) 
    if k == 0: 
        kernel_size = 2
    elif k == 1:
        kernel_size = 3
    elif k == 2:
        kernel_size = 5
    else:
        kernel_size = 11
    
    max_lags = ((2**num_conv)*kernel_size)*pilhas
    min_lags = (2**num_conv)
  
    return genotypeCNN3(
        pilhas,
        random.randint(0, 2), 
        random.uniform(0, 0.5), 
        random.randint(0, 1),  
        random.randint(min_lags, max_lags),  
        num_conv,
        kernel_size,
        2
    )

def genotypeCNN1(filters, pool, dropout, norm, lags, num_conv, kernel_size, pool_size, tipo):
    """
    Cria o genótipo para um indivíduo CNN
    :parametro filters: número de filtros [16,32,64]
    :parametro pool: probabilidade de pooling [0%, 50%, 100%] 
    :parametro dropout: porcentagem dropout (0.5 a 0.8)
    :parametro norm: normalização (0 - sim, 1 - não)
    :parametro lags: quantidade de lags [1 - 50]
    :parametro num_conv: número de camadas convolucionais [1 - 6]
    :parametro kernel_size: tamanho do kernel de convolução [2,3,5,11]
    :parametro pool_size: tamanho da janela de pooling [2,3,4,5]
    :parametro tipo: tipo de ensemble (0-CNN1, 1-CNN2, 2-CNN3, 3-Híbrido)
    :return: o genótipo, um dicionário com todos os hiperparâmetros
    """
    ind = {
        'filters': filters, 
        'pool': pool, 
        'dropout': dropout, 
        'norm': norm,
        'lags': lags,
        'num_conv': num_conv,
        'kernel_size': kernel_size,
        'pool_size': pool_size,
        'tipo': tipo 
        }
    return ind

def genotypeCNN2(filters, dropout, norm, lags, num_conv, kernel_size, tipo):
    """
    Cria o genótipo para um indivíduo CNN2
    :parametro filters: número de filtros [16,32,64]
    :parametro dropout: porcentagem dropout (0 a 0.5)
    :parametro norm: normalização (0 - sim, 1 - não)
    :parametro lags: quantidade de lags em relação ao número de convoluções e tamanho do kernel
    :parametro num_conv: número de camadas convolucionais [1 - 5]
    :parametro kernel_size: tamanho do kernel de convolução [2,3,5,11]
    :parametro tipo: tipo de ensemble (0-CNN1, 1-CNN2, 2-CNN3, 3-Híbrido)
    :return: o genótipo, um dicionário com todos os hiperparâmetros
    """
    ind = {
      'filters': filters,
      'dropout': dropout, 
      'norm': norm,
      'lags': lags,
      'num_conv': num_conv,
      'kernel_size': kernel_size,
      'tipo': tipo

    }
    return ind

def genotypeCNN3(blocos, filters, dropout, norm, lags, num_conv, kernel_size, tipo):
    """
    Cria o genótipo para um indivíduo CNN3
    :parametro pilhas: número de blocos [1,2]
    :parametro filters: número de filtros [16,32,64]
    :parametro dropout: porcentagem dropout (0 a 0.5)
    :parametro norm: normalização (0 - sim, 1 - não)
    :parametro lags: quantidade de lags em relação ao número de convoluções e tamanho do kernel
    :parametro num_conv: número de camadas convolucionais [1 - 5]
    :parametro kernel_size: tamanho do kernel de convolução [2,3,5,11]
    :parametro tipo: tipo de ensemble (0-CNN1, 1-CNN2, 2-CNN3, 3-Híbrido)
    :return: o genótipo, um dicionário com todos os hiperparâmetros
    """
    ind = {
      'pilhas': blocos,
      'filters': filters, 
      'dropout': dropout,
      'norm': norm,
      'lags': lags, 
      'num_conv': num_conv, 
      'kernel_size': kernel_size, 
      'tipo': tipo
    }
    return ind
    
def slideWindow(train, test, n_lags):
    """
    Separa os dados de treinamento e teste
    parametro train: dados de treinamento
    parametro test: dados de teste
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(n_lags, len(train)):
        X_train.append(train[i-n_lags:i])
        y_train.append(train[i])
    for i in range(n_lags, len(test)):
        X_test.append(test[i-n_lags:i])
        y_test.append(test[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    return X_train, y_train, X_test, y_test

def slideWindowMulti(train, test, n_lags, n_var):
    """
    Separa os dados de treinamento e teste
    parametro train: dados de treinamento
    parametro test: dados de teste
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # print(f'\t[slideWindowMulti]: train.shape = {train.shape}, test.shape = {test.shape}')

    for i in range((n_var*n_lags), len(train)-n_var+1, n_var):
        X_train.append(train[i-(n_var*n_lags):i])
        y_train.append(train[i+n_var-1])
    for i in range((n_var*n_lags), len(test)-n_var+1, n_var):
        X_test.append(test[i-(n_var*n_lags):i])
        y_test.append(test[i+n_var-1])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    return X_train, y_train, X_test, y_test

def modelo_CNN3(X_train, y_train, individual):
    """
    Cria um modelo CNN3
    :parametro X_train: dados para treinamento
    :parametro y_train: rótulo dos dados de treinamento
    :parametro individual: dicionário com os hiperparâmetros do modelo
    :return: o modelo
    """
    warnings.filterwarnings('ignore')
    call = [EarlyStopping(monitor='loss', mode='min', patience=15, verbose=1),]
    if individual['filters'] == 0: 
        filters = 16
    elif individual['filters'] == 1:
        filters = 32
    else:
        filters = 64
    
    if individual['norm'] == 0:
        norm = False
    else:
        norm = True
  
    if individual['kernel_size'] == 0: 
        kernel_size = 2
    elif individual['kernel_size'] == 1:
        kernel_size = 3
    elif individual['kernel_size'] == 2:
        kernel_size = 5
    else:
        kernel_size = 11
    
    d = []
    for i in range(individual['num_conv']):
        d.append(2**i)
    i = Input(batch_shape=(None,X_train.shape[1],1))
    o = TCN(nb_filters=filters, kernel_size=kernel_size, nb_stacks=individual['pilhas'], dilations=d,
    padding='causal', use_skip_connections=False, dropout_rate=individual['dropout'], return_sequences=False, name='tcn')(i)
        
    o = Dense(1)(o)
    model = Model(inputs=[i], outputs=[o])
    model.compile(optimizer='Adam', loss='mse')  
    history = model.fit(X_train, y_train, epochs = 30, verbose=0, batch_size = filters, callbacks = call)      
  
    return model, history

def modelo_CNN1(X_train, y_train, individual):
    """
    Cria um modelo CNN1
    :parametro X_train: dados para treinamento
    :parametro y_train: rótulo dos dados de treinamento
    :parametro individual: dicionário com os hiperparâmetros do modelo
    :return: o modelo
    """
    warnings.filterwarnings('ignore')
    call = [EarlyStopping(monitor='loss', mode='min', patience=15, verbose=1),]
    model = Sequential()
  
    if individual['filters'] == 0: 
        filters = 16
    elif individual['filters'] == 1:
        filters = 32
    else:
        filters = 64
  
    if individual['kernel_size'] == 0: 
        kernel_size = 2
    elif individual['kernel_size'] == 1:
        kernel_size = 3
    elif individual['kernel_size'] == 2:
        kernel_size = 5
    else:
        kernel_size = 11
    
    try:
        for i in range(individual['num_conv']):
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1],1),
                                padding='same', kernel_constraint=max_norm(3)))
            if individual['pool'] == 0: 
                model.add(MaxPooling1D(pool_size=individual['pool_size'], strides=2, padding='same', data_format='channels_first'))
            elif individual['pool'] == 1:
                rnd = random.uniform(0,1)
                if rnd > .5:
                    model.add(MaxPooling1D(pool_size=individual['pool_size']))
            model.add(SpatialDropout1D(round(individual['dropout'],1)))
            if individual['norm'] == 0:
                model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(1, activation = 'linear'))
        model.compile(loss='mse', optimizer='Adam')
        history = model.fit(X_train, y_train, epochs = 100, verbose=0, batch_size = filters, callbacks = call)
        
        return model, history
    except Exception as ex:
        individual = random_CNN1()
        model, history = modelo_CNN1(X_train, y_train, individual)
        return model, history
    
def modelo_CNN2(X_train, y_train, individual):
    """
    Cria um modelo CNN2
    :parametro X_train: dados para treinamento
    :parametro y_train: rótulo dos dados de treinamento
    :parametro individual: dicionário com os hiperparâmetros do modelo
    :return: o modelo
    """
    warnings.filterwarnings('ignore')
    model = Sequential()
    call = [EarlyStopping(monitor='loss', mode='min', patience=15, verbose=1),]
    if individual['filters'] == 0: 
        filters = 16
    elif individual['filters'] == 1:
        filters = 32
    else:
        filters = 64
  
    if individual['kernel_size'] == 0: 
        kernel_size = 2
    elif individual['kernel_size'] == 1:
        kernel_size = 3
    elif individual['kernel_size'] == 2:
        kernel_size = 5
    else:
        kernel_size = 11
    
    for i in range(individual['num_conv']):
        d = 2**i
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1],1),
                             padding='causal',dilation_rate = d, kernel_constraint=max_norm(3)))
        model.add(SpatialDropout1D(round(individual['dropout'],1)))
        if individual['norm'] == 0:
            model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='Adam')
    history = model.fit(X_train, y_train, epochs = 60, verbose=0, batch_size = filters, callbacks = call)          
    return model, history
    
def slideWindow_val(series, n_lags):
    """
    Slide window para dados de validação
    :parametro series: base de dados
    :parametro n_lags: número de lags
    :return: base de dados separada com rótulos
    """
    X_test = []
    y_test = []
    for i in range(n_lags, len(series)):
        X_test.append(series[i-n_lags:i])
        y_test.append(series[i])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    X_test = X_test.astype ('float32')
    return X_test, y_test

def slideWindowMulti_val(series, n_lags, n_var):
    """
    Slide window para dados de validação
    :parametro series: base de dados
    :parametro n_lags: número de lags
    :return: base de dados separada com rótulos
    """
    X_test = []
    y_test = []
    # for i in range(n_lags, len(series)):
    for i in range(((n_var*n_lags)), len(series)-n_var+1, n_var):
        X_test.append(series[i-(n_var*n_lags):i])
        y_test.append(series[i+n_var-1])
        # y_test.append(series[i+n_var-1])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    X_test = X_test.astype ('float32')
    return X_test, y_test

def predictModel(series, model,n_previsoes, n_lags, scaler):
    """
    Faz a previsão dos modelos
    :parametro series: base de dados
    :parametro model: modelo
    :parametro n_lags: número de lags
    :parametro scaler: escala para normalização 
    :return: rmse da previsão, yhat: valor previsto, y_test: valor real
    """
    X_test, y_test = slideWindow_val(series, n_lags)
    yhat = np.zeros((y_test.shape[0],n_previsoes))
    rmse = []
    mape = []
    for i in range(len(X_test)):
        X = X_test[i,:,0].reshape((1, X_test.shape[1], X_test.shape[2]))
        for j in range(n_previsoes):
            yhat[i,j] = model.predict(X, verbose=0)
            X = np.insert(X,n_lags,yhat[i,j],axis=1) 
            X = np.delete(X,0,axis=1)
    yhat = scaler.inverse_transform(yhat)
    y_test = scaler.inverse_transform(y_test)
    for i in range(n_previsoes):
        rmse.append(np.sqrt(mean_squared_error(yhat[:,i],y_test[:])))
    return rmse, yhat, y_test

# basic.predictModelMulti(test, model, n_previsoes=10, n_lags=2*star_CNN1['lags'], scaler=m_scaler)
def predictModelMulti(series, model, n_previsoes, n_lags, n_var, scaler):
    """
    Faz a previsão dos modelos
    :parametro series: base de dados
    :parametro model: modelo
    :parametro n_lags: número de lags
    :parametro scaler: escala para normalização 
    :return: rmse da previsão, yhat: valor previsto, y_test: valor real
    """
    X_test, y_test = slideWindowMulti_val(series, n_lags, n_var)
    yhat = np.zeros((y_test.shape[0],n_previsoes))
    rmse = []
    mape = []
    # print(f'predictModelMulti')
    # print(f'n_var = {n_var}, n_lags = {n_lags}, len(test) = {len(X_test)}, n_previsoes = {n_previsoes}')
    for i in range(len(X_test)):
    # for i in range(len(X_test)-n_var+1):
        X = X_test[i,:,0].reshape((1, X_test.shape[1], X_test.shape[2]))
        for j in range(n_previsoes):
            # print(f'i, j = {i}, {j}')
            yhat[i,j] = model.predict(X, verbose=0)
            # X = np.insert(X,n_lags,yhat[i,j],axis=1)
            X = np.insert(X,n_var*n_lags,yhat[i,j],axis=1)
            X = np.delete(X,0,axis=1)
    yhat = scaler.inverse_transform(yhat)
    y_test = scaler.inverse_transform(y_test)
    for i in range(n_previsoes):
        rmse.append(np.sqrt(mean_squared_error(yhat[:,i],y_test[:])))
    return rmse, yhat, y_test
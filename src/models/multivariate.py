# Imports

import os
import time
import gc
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from statsmodels.tsa.api import VAR

from utils import basic
from utils import Ensemble as es

import torch
from torch import nn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = remove INFO, 2 = remove INFO + WARNING, 3 = only ERROR


# Function definitions

def normalize(df):
    mindf = df.min()
    maxdf = df.max()
    return (df-mindf)/(maxdf-mindf)


def denormalize(norm, _min, _max):
    return [(n * (_max-_min)) + _min for n in norm]


def to_tensor(data, features, target):
    X = torch.tensor(data[features].values, dtype=torch.float32)
    y = torch.tensor(data[target].values, dtype=torch.float32)
    return X, y    


class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, num_layers, hidden_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.transformer = nn.Transformer(d_model=input_dim, nhead=nhead, num_encoder_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, src):
        # src shape: (sequence_length, batch_size, input_dim)
        transformer_out = self.transformer(src, src)
        # Take the last output (for prediction)
        out = self.fc(transformer_out[-1, :, :])
        return out


def train_model(model, X_train, y_train, num_epochs, batch_size, optimizer, criterion):
    model.train()
    for epoch in range(num_epochs):
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            output = model(batch_X.unsqueeze(1))  # reshape to (sequence_length, batch_size, input_dim)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()


def evaluate_model(model, X_test, y_test, criterion):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.unsqueeze(1)).squeeze()
        mse = criterion(predictions, y_test)
        return predictions, mse.item()


def add_lag_features(df, features, lag):
    for feature in features:
        for i in range(1, lag + 1):
            df[f'{feature}_lag_{i}'] = df[feature].shift(i)
    df.dropna(inplace=True)
    return df


def get_search_dataset_multivariate_for_var(dataset, n_var, vars, num_lags=4, n_splits=5):
    df1 = pd.read_csv(dataset, sep=";")
    df1['Tamp'] = (df1['Tmax'] - df1['Tmin'])/2
    df1 = df1[vars]
    
    ints = df1.select_dtypes(include=['int64','int32','int16']).columns
    df1[ints] = df1[ints].apply(pd.to_numeric, downcast='integer')
    floats = df1.select_dtypes(include=['float']).columns
    df1[floats] = df1[floats].apply(pd.to_numeric, downcast='float')
    
    series = df1
    norm_df = normalize(series)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    return norm_df, tscv


def get_search_dataset_multivariate_for_rf(dataset, vars):
    df1 = pd.read_csv(dataset, sep=";")
    df1['Tamp'] = (df1['Tmax'] - df1['Tmin'])/2
    num_lags = 4

    X = df1[vars[:-1]]
    y = df1['ETo']

    for var in vars:
        for lag in range(1, num_lags + 1):
            X[f'{var}_lag{lag}'] = df1[var].shift(lag)

    X = X.dropna()
    y = y.loc[X.index]
    
    n_split = 5
    tscv = TimeSeriesSplit(n_split)
    
    return X, y, tscv


def get_search_dataset_multivariate_for_tcnn(dataset, n_var, vars, num_lags=4, n_splits=5):
    df1 = pd.read_csv(dataset, sep=";")
    df1['Tamp'] = (df1['Tmax'] - df1['Tmin'])/2
    df1 = df1[vars]
    
    # Downcast nos tipos
    ints = df1.select_dtypes(include=['int64', 'int32', 'int16']).columns
    df1[ints] = df1[ints].apply(pd.to_numeric, downcast='integer')
    floats = df1.select_dtypes(include=['float']).columns
    df1[floats] = df1[floats].apply(pd.to_numeric, downcast='float')
    
    #series = df1.iloc[:, 1:n_var+1]
    series = df1
    norm_df = normalize(series)

    # Aqui fazemos o split com base no TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    return norm_df, tscv


def get_search_dataset_multivariate_for_transformer(dataset, vars, lag=4, n_splits=5):
    df1 = pd.read_csv(dataset, sep=";")
    df1['Tamp'] = (df1['Tmax'] - df1['Tmin']) / 2.
    features = vars
    target = 'ETo'

    df1 = add_lag_features(df1, features, lag)

    scaler = StandardScaler()
    scaled_features = [f'{feature}_lag_{i}' for feature in features for i in range(1, lag + 1)]
    df1[scaled_features] = scaler.fit_transform(df1[scaled_features])

    X = df1[scaled_features]
    y = df1[target]
        
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    input_dim = len(scaled_features)
    
    return X, y, tscv, input_dim


def form_data(data, t, n_execucoes, n_previsoes):
    df = pd.DataFrame(data)
    df1 = df.T
    frames = [df1.iloc[:,0], df1.iloc[:,1], df1.iloc[:,2], df1.iloc[:,3], df1.iloc[:,4], df1.iloc[:,5], df1.iloc[:,6], df1.iloc[:,7], df1.iloc[:,8], df1.iloc[:,9], df1.iloc[:,10], df1.iloc[:,11],
          df1.iloc[:,12], df1.iloc[:,13], df1.iloc[:,14], df1.iloc[:,15], df1.iloc[:,16], df1.iloc[:,17],df1.iloc[:,18], df1.iloc[:,19], df1.iloc[:,20], df1.iloc[:,21], df1.iloc[:,22], 
          df1.iloc[:,23], df1.iloc[:,24], df1.iloc[:,25], df1.iloc[:,26], df1.iloc[:,27], df1.iloc[:,28], df1.iloc[:,29]]
    result = pd.concat(frames)
    r = pd.DataFrame(result) 
    r.insert(1, "Modelo", True)
    for i in range(n_execucoes * n_previsoes): # n_execucoes * n_previsoes
        r['Modelo'].iloc[i] = t
    return r


def run_var_model(dataset_file_name, result_file_name, sufix, n_var, n_execucoes, n_previsoes, vars):
    results = []
    df, tscv = get_search_dataset_multivariate_for_var(dataset_file_name, n_var, vars=vars)
    
    for i in range(n_execucoes):
        rmse_scores = []
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]

            model = VAR(train)
            model_fit = model.fit(maxlags=4)
            lag_order = model_fit.k_ar

            if len(test) <= lag_order:
                continue  # pula se o test for menor que a ordem do lag

            fc = model_fit.forecast(y=train.values[-lag_order:], steps=len(test))
            predictions = pd.DataFrame(fc, index=test.index, columns=train.columns)

            # Calcula RMSE apenas para a variável alvo (ex: 'ETo')
            rmse = np.sqrt(mean_squared_error(test['ETo'], predictions['ETo']))
            rmse_scores.append(rmse)

        mean_rmse = np.mean(rmse_scores)
        results.append(mean_rmse)
        # print(f"[{i + 1}-ésima execução] Média do RMSE em todos os splits: {mean_rmse:.6f}")
        #print(f".")

    results = form_data(results, sufix, n_execucoes, n_previsoes)
    results.to_csv(result_file_name, index=True)

    del df
    del tscv
    gc.collect()


def run_rf_model(dataset_file_name, result_file_name, sufix, n_execucoes, n_previsoes, vars):
    results = []
    
    X, y, tscv = get_search_dataset_multivariate_for_rf(dataset_file_name, vars=vars)
      
    for i in range(n_execucoes):
        
        rmse_scores = []
        model = RandomForestRegressor(n_estimators=100)#, random_state=42)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_scores.append(rmse)
            
        mean_rmse = np.mean(rmse_scores)
        results.append(mean_rmse)        
        # print(f"[{i + 1}-ésima execução] Média do RMSE em todos os splits: {mean_rmse:.6f}")
        #print(f".")

    results = form_data(results, sufix, n_execucoes, n_previsoes)
    results.to_csv(result_file_name,index=True)

    del X
    del y
    del tscv
    gc.collect()


def run_tcnn_model(dataset_file_name, result_file_name, sufix, star, n_var, vars, n_execucoes, n_previsoes):
    df, tscv = get_search_dataset_multivariate_for_tcnn(dataset_file_name, n_var=n_var, vars=vars)

    results = []

    for i in range(n_execucoes):
        rmse_scores = []
        
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]

            # Pré-processamento e criação dos conjuntos para o modelo
            train, test, scaler = es.get_dados(star, train, test)
            X_train, y_train, X_test, y_test = basic.slideWindowMulti(train, test, n_lags=star['lags'], n_var=n_var)

            model, _ = basic.modelo_CNN1(X_train, y_train, star)
            rmse, yhat, y_test = basic.predictModelMulti(test, model, n_previsoes=n_previsoes, n_lags=star['lags'], n_var=n_var, scaler=scaler)
            
            rmse_scores.append(rmse[0])  # Assumindo que rmse é uma lista ou array com [ETo]

        mean_rmse = np.mean(rmse_scores)
        results.append(mean_rmse)        
        # print(f"[{i + 1}-ésima execução] Média do RMSE em todos os splits: {mean_rmse:.6f}")
        #print(f".")
    
    results = form_data(results, sufix, n_execucoes, n_previsoes)
    results.to_csv(result_file_name, index=True)
    
    del df
    del tscv
    gc.collect()


def run_tfts_model(dataset_file_name, result_file_name, sufix, n_execucoes, n_previsoes, vars):
    nhead = 2
    num_layers = 2
    hidden_dim = 64
    output_dim = 1
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 32

    results = []
    
    X, y, tscv, input_dim = get_search_dataset_multivariate_for_transformer(dataset_file_name, vars=vars)
    
    for i in range(n_execucoes):
        
        rmse_scores = []
        model = TransformerModel(input_dim, nhead, num_layers, hidden_dim, output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        for train_index, test_index in tscv.split(X):            
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            X_train, y_train = torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32)
            X_test, y_test = torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32)
            
            train_model(model, X_train, y_train, num_epochs, batch_size, optimizer, criterion)
            predictions, mse = evaluate_model(model, X_test, y_test, criterion)
            rmse = np.sqrt(mse)
            rmse_scores.append(rmse)
            
        mean_rmse = np.mean(rmse_scores)
        results.append(mean_rmse)
        # print(f"[{i + 1}-ésima execução] Média do RMSE em todos os splits: {mean_rmse:.6f}")
        #print(f".")
        
    results = form_data(results, sufix, n_execucoes, n_previsoes)
    results.to_csv(result_file_name,index=True)

    del nhead
    del num_layers
    del hidden_dim
    del output_dim
    del learning_rate
    del num_epochs
    del batch_size
    del X
    del y
    del tscv
    del input_dim
    gc.collect()


# Parameters for all the models

n_var=7
n_lags=4
n_execucoes=30
n_previsoes=1

# Parameters for TCNN

star_CNN1 = {'filters': 1, 'pool': 0, 'pool_size': 3, 'dropout': 0.012594059561340142, 'norm': 1, 'lags': 4, 'num_conv': 1, 'kernel_size': 3, 'rmse': 0.7696852129001718, 'num_param': 449}
# star_CNN2 = {'filters': 1, 'dropout': 0, 'norm': 1, 'lags': 4, 'num_conv': 1, 'kernel_size': 0, 'rmse': 0.7566198577347709, 'num_param': 449}
# star_CNN3 = {'pilhas': 2, 'filters': 1, 'dropout': 0.2, 'norm': 1, 'lags': 48, 'num_conv': 3, 'kernel_size': 2, 'rmse': 0.7530, 'num_param': 68257}


project_root = Path(__file__).resolve().parents[2]  # Sobe até a raiz: C:\my_tcc_project
bases_path = project_root / 'data' / 'processed'

prefixes = ['important'] # ['all', 'important']

for prefix in prefixes:    
    # vars = ['Rs', 'u2', 'Tmax', 'Tmin', 'RH', 'pr', 'ETo']  # original
    if prefix == 'all':
        vars = ['Rs', 'u2', 'Tmax', 'Tmin', 'RH', 'ETo']  # all
    elif prefix == 'important':
        #vars = ['Rs', 'Tmax']  # selected
        #vars = ['Rs', 'Tmax', 'ETo']  # selected
        #vars = ['Rs', 'Tamp', 'u2', 'ETo']  # selected
        #vars = ['Rs', 'Tmax', 'Tmin', 'ETo']  # selected
        vars = ['Rs', 'Tmax', 'u2', 'ETo']  # selected

    models = ['VAR', 'RF', 'TCNN']

    # Running the models

    for m in models:
        if m == 'VAR':
            sufix='VAR ('
            for v in vars:
                sufix += str(v) + ', '
            sufix += ')'
            sufix = sufix.replace(', )',')')
            model_type = 'VAR'
            model_title = 'Vector Autoregressive'
        elif m == 'RF':        
            sufix='RF ('
            for v in vars:
                sufix += str(v) + ', '
            sufix += ')'
            sufix = sufix.replace(', )',')')
            model_type = 'RF'
            model_title = 'Random Forest'
        elif m == 'TCNN':
            sufix='TCNN ('
            for v in vars:
                sufix += str(v) + ', '
            sufix += ')'
            sufix = sufix.replace(', )',')')
            model_type = 'TCNN'
            model_title = 'Temporal Convolutional Neural Network'
        elif m == 'TFTS':
            sufix='TFTS ('
            for v in vars:
                sufix += str(v) + ', '
            sufix += ')'
            sufix = sufix.replace(', )',')')
            model_type = 'TFTS'
            model_title = 'Transfomer for Time Series'
        else:
            break
        for dataset in os.listdir(bases_path):
            if dataset.endswith('.csv'):
                result_path = f'results/{prefix}_variables/{prefix}_vars_results_{model_type}_{str(n_execucoes)}_{str(n_previsoes)}_{str(dataset)}'
                if prefix == 'important':
                    aux = f'vars_'+str(vars)+'_results'
                    aux = aux.replace("'", "").replace("[", "").replace("]", "").replace(", ", "")
                    result_path = result_path.replace("vars_results", aux)
                dataset_path = os.path.join(bases_path, dataset)
                print(f'\n\nRunning {model_title} [from multivariate.py]: base = {dataset_path}\n.')
                result_path = project_root / result_path
                start = time.time()
                print(f'Started at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))}...')
                if m == 'VAR':
                    run_var_model(dataset_file_name=str(dataset_path), result_file_name=result_path, sufix=sufix, n_var =n_var, n_execucoes=n_execucoes, n_previsoes=n_previsoes, vars=vars)
                elif m == 'RF':
                    run_rf_model(dataset_file_name=str(dataset_path), result_file_name=result_path, sufix=sufix, n_execucoes=n_execucoes, n_previsoes=n_previsoes, vars=vars)
                elif m == 'TCNN':
                    run_tcnn_model(dataset_file_name=str(dataset_path), result_file_name=result_path, sufix=sufix, star=star_CNN1, n_var =n_var, vars=vars, n_execucoes=n_execucoes, n_previsoes=n_previsoes)
                elif m == 'TFTS':
                    run_tfts_model(dataset_file_name=str(dataset_path), result_file_name=result_path, sufix=sufix, n_execucoes=n_execucoes, n_previsoes=n_previsoes, vars=vars)
                else:
                    break
                stop = time.time()
                segundos = int(stop - start)
                minutos = int(segundos / 60)
                horas = int(minutos / 60)
                print(f'...done! Execution time = {int(int(int(stop - start) / 60) / 60)}h {int(int(stop - start) / 60)%60}m {int(stop - start)%60}s.')

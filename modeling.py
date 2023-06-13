import pandas as pd
from scipy.signal import savgol_filter, filtfilt, butter
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from typing import Dict, List, Tuple
import logging
import modeling_settings as mds


def save_models(models):
    model_filename = 'models_sklearn.pkl'
    sklearn_models = {}
    models_list = []
    for model_name, model in models[0].items():
        if model_name in mds.TS_models_list:
            models_list.append(model_name)
            ts_filename = f'{model_name}.h5'
            model.save(ts_filename)
        else:
            sklearn_models[model_name] = model
    with open(model_filename, 'wb') as file:
        pickle.dump(sklearn_models, file)
    with open('models_list.pkl', 'wb') as file:
        pickle.dump(models_list, file)
    with open('models_metrics.pkl', 'wb') as file:
        pickle.dump(models[1], file)
    with open('models_scores.pkl', 'wb') as file:
        pickle.dump(models[2], file)
    with open('columns_list.pkl', 'wb') as file:
        pickle.dump(models[3], file)


def load_models():
    with open('models_list.pkl', 'rb') as file:
        models_list = pickle.load(file)
    model_filename = 'models_sklearn.pkl'
    with open(model_filename, 'rb') as file:
        models = pickle.load(file)
    for model_name in mds.TS_models_list:
        ts_filename = f'{model_name}.h5'
        model = keras.models.load_model(ts_filename)
        models[model_name] = model
    return models


def preprocessing_trains(trains):
    trains['start_month'] = pd.to_datetime(trains['start_id'].str[0:10], format='%d.%m.%Y').dt.month
    trains['start_day'] = pd.to_datetime(trains['start_id'].str[0:10], format='%d.%m.%Y').dt.day
    trains['update_month'] = trains['update'].dt.month
    trains['update_day'] = trains['update'].dt.day
    trains.dropna(subset=['start', 'ops station', 'o_road', 'to_home'], inplace=True)
    trains.loc[trains['o_road'] == 'ЛИТОВСКАЯ', 'o_road'] = '(99)'
    trains['o_road'] = trains['o_road'].str[-3:-1].astype(int)
    trains['start'] = trains['start'].str[-7:-1].astype(int)
    trains['ops station'] = trains['ops station'].str[-7:-1].astype(int)
    return trains


def preprocessing_update_trains(update_trains):
    update_trains.dropna(subset=['train_start', 'расстояние до Лены', 'op_station_index', 'ops station'], inplace=True)
    update_trains.train_start = update_trains.train_start.astype(int)
    update_trains['start_month'] = pd.to_datetime(update_trains['start_id'].str[0:10], format='%d.%m.%Y').dt.month
    update_trains['start_day'] = pd.to_datetime(update_trains['start_id'].str[0:10], format='%d.%m.%Y').dt.day
    update_trains['update_month'] = update_trains['update'].dt.month
    update_trains['update_day'] = update_trains['update'].dt.day
    update_trains['DLeft'] = update_trains['расстояние до Лены'].astype(int)
    update_trains['ops station'] = update_trains['ops station'].str.extract(r'\((\d+)\)').astype(float)
    update_trains.dropna(subset=['ops station'], inplace=True)
    update_trains['ops station'] = update_trains['ops station'].astype(int)
    update_trains['start'] = update_trains['train_start'].astype(int)
    update_trains.loc[update_trains['ops road'] == 'ЛИТОВСКАЯ', 'ops road'] = '(99)'
    update_trains.dropna(subset=['ops road'], inplace=True)
    update_trains['o_road'] = update_trains['ops road'].str[-3:-1].astype(int)
    update_trains['ops station'].astype('string')
    return update_trains


def smooth_data(data, filter_type=None):
    if filter_type == 'savgol':
        window_size = 100
        poly_order = 1
        smooth_data = savgol_filter(data, window_size, poly_order)
    elif filter_type == 'butter':
        cutoff_freq = 0.001
        filter_order = 5
        b, a = butter(filter_order, cutoff_freq, btype='low', analog=False, output='ba')
        smooth_data = filtfilt(b, a, data)
    else:
        smooth_data = data

    return smooth_data


def cross_validation_test(models, X, y):
    scoring_metric = 'neg_mean_squared_error'
    num_folds = 5
    scores = {}
    for name, model in models.items():
        if name in mds.TS_models_list:
            logging.info('TensorFlow models, no scoring')
            scores[name] = [0, 0]
        else:
            logging.info(f'scoring for model {name} started')
            cross_scores = cross_val_score(estimator=model, X=X, y=y, cv=num_folds, scoring=scoring_metric)
            scores[name] = [-cross_scores.mean(), cross_scores.std()]
            logging.info(f'scoring for model {name} finished')
    scores = pd.DataFrame(scores)
    scores.index = ['Mean', 'Std']
    return scores


def get_models_metrics(models, X_test, y_test):
    models_metrics = {}
    for name, model in models.items():
        logging.info(f'predicting for model {name} started')
        y_pred = model.predict(X_test)
        mae, mse, rmse = mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred), mean_squared_error(y_test, y_pred, squared=False)
        models_metrics[name] = [mae, mse, rmse]
        logging.info(f'predicting for model {name} finished')
    metrics = pd.DataFrame(models_metrics)
    metrics.index = ['MAE', 'MSE', 'RMSE']
    return metrics


def create_models(df, columns_list):
    logging.info('Started preprocessing')
    trains = preprocessing_trains(df)
    logging.info('Preprocessing done')
    smoothing_factor = 'savgol'
    trains['target'] = smooth_data(trains['to_home'], smoothing_factor)
    logging.info('Filtering done')
    X = trains[columns_list]
    y = trains['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info('Data setup done')
    logging.info('Fitting models')
    models = {'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42, max_depth=100),
              'DecisionTree': DecisionTreeRegressor(max_depth=1000, random_state=42),
              'KNeighbors': KNeighborsRegressor(n_neighbors=5),
              'ExtraTrees': ExtraTreesRegressor(random_state=42, n_estimators=300, max_depth=100),
              'GradientBoosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.75, random_state=42),
              'TensorFlow_Elu_Rmsprop': keras.Sequential([layers.Dense(128, activation='elu', input_shape=[len(columns_list)]),
                                                          layers.Dense(128, activation='elu'), 
                                                          layers.Dense(1)])
                                                          ,
              'TensorFlow_Softplus_Nadam': keras.Sequential([layers.Dense(128, activation='softplus', input_shape=[len(columns_list)]),
                                                             layers.Dense(128, activation= 'softplus'),
                                                             layers.Dense(1)])
                                                             ,
              'TensorFlow_Synthetic': keras.Sequential([layers.Dense(128, activation='elu', input_shape=[len(columns_list)]),
                                                        layers.Dense(128, activation='relu'), 
                                                        layers.Dense(128, activation='softplus'),
                                                        layers.Dense(128, activation='softplus'), 
                                                        layers.Dense(1)])}
    fit_models = {}
    for name, model in models.items():
        logging.info(f'fitting model {name} started')
        if name == 'TensorFlow_Elu_Rmsprop' or name == 'TensorFlow_Synthetic':
            model.compile(loss='mean_squared_error', optimizer=keras.optimizers.RMSprop(learning_rate=0.001))
            keras.optimizers.RMSprop(learning_rate=0.001).build(model.trainable_variables)
            model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test)) 
            fit_models[name] = model
        elif name == 'TensorFlow_Softplus_Nadam':
            model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Nadam(learning_rate=0.001))
            keras.optimizers.Nadam(learning_rate=0.001).build(model.trainable_variables)
            model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
            fit_models[name] = model
        else:         
            fit_models[name] = model.fit(X_train, y_train)
        logging.info(f'fitting model {name} finished')
    logging.info('All models fitted')
    logging.info('Calculating metrics')
    metrics = get_models_metrics(models, X_test, y_test)
    logging.info('Metrics calculated')
    logging.info(f"Metrics: \n{metrics.to_string(index=True, line_width=80)}")
    logging.info('Calculating scores')
    scores = cross_validation_test(models, X_test, y_test)
    logging.info('Scores calculated')
    logging.info(f"Scores: \n{scores.to_string(index=True, line_width=80)}")
    return [fit_models, metrics, scores, columns_list]


def prediction(df, models, columns_list):
    logging.info('Started preprocessing')
    update_trains = preprocessing_update_trains(df)
    logging.info('Preprocessing done')
    update_X = update_trains[columns_list]
    logging.info('Predicting')
    columns_to_keep = []
    for name, model in models.items():
        logging.info(f'predicting for model {name} started')
        update_Y = model.predict(update_X)
        logging.info(f'predicting for model {name} finished')
        logging.info(update_Y)
        logging.info(type(update_Y))
        duration = 'duration_' + name
        update_trains[duration] = pd.DataFrame(update_Y)#.astype(float)
        expected_delivery = 'expected_delivery_' + name
        update_trains[expected_delivery] = pd.to_datetime(update_trains['update']) + pd.to_timedelta(update_trains[duration], unit='D')
        update_trains[expected_delivery] = update_trains[expected_delivery]
        update_trains = update_trains.sort_values(expected_delivery)
        columns_to_keep.append(duration)
        columns_to_keep.append(expected_delivery)
    logging.info('Predicting done')
    columns_to_keep.append('update')
    columns_to_keep.append('котлов')
    columns_to_keep.append('_num')
    update_trains = update_trains[columns_to_keep]
    return update_trains


if __name__ == "__main__":
    pass

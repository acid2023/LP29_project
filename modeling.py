import pandas as pd
from scipy.signal import savgol_filter, filtfilt, butter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
import tensorflow as tf
import pickle
from typing import Dict, List
import logging
import modeling_settings as mds
import folders
from folders import folder_check
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import osm
import numpy as np


def save_models(models_dict: Dict) -> None:
    path = folder_check(folders.models_folder)
    model_filename = f'{path}models_sklearn.pkl'
    sklearn_models = {}
    models_list = []
    for model_name, model in models_dict['fit_models'].items():
        if model_name in mds.TF_models_list:
            models_list.append(model_name)
            ts_filename = f'{path}{model_name}.h5'
            model.save(ts_filename)
        else:
            sklearn_models[model_name] = model
    with open(model_filename, 'wb') as file:
        pickle.dump(sklearn_models, file)
    with open(f'{path}scalers.pkl', 'wb') as file:
        pickle.dump(models_dict['scalers'], file)
    with open(f'{path}models_list.pkl', 'wb') as file:
        pickle.dump(models_list, file)
    with open(f'{path}models_metrics.pkl', 'wb') as file:
        pickle.dump(models_dict['metrics'], file)
    with open(f'{path}models_scores.pkl', 'wb') as file:
        pickle.dump(models_dict['scores'], file)
    with open(f'{path}columns.pkl', 'wb') as file:
        pickle.dump(models_dict['columns'], file)
    with open(f'{path}encoders.pkl', 'wb') as file:
        pickle.dump(models_dict['encoders'], file)


def load_models() -> Dict:
    path = folder_check(folders.models_folder)
    model_filename = f'{path}models_sklearn.pkl'
    with open(f'{path}columns', 'rb') as file:
        columns = pickle.load(file)
    with open(model_filename, 'rb') as file:
        models = pickle.load(file)
    with open(f'{path}models_list.pkl', 'wb') as file:
        models_list = pickle.load(file)
    for model_name in models_list:
        if model_name in mds.TF_models_list:
            models[model_name] = keras.models.load_model(f'{path}{model_name}.h5')
    scalers = pickle.load(open(f'{path}scalers.pkl', 'rb'))
    encoders = pickle.load(open(f'{path}encoders.pkl', 'rb'))
    return {'models': models, 'scalers': scalers, 'encoders': encoders, 'columns': columns}



def preprocessing_trains(df: pd.DataFrame) -> List:
    logging.info('starting coding stations')
    df = df.replace({'': np.nan, 'NA': np.nan, 'None': np.nan})
    df = df.dropna(subset=['ops station', 'o_road', 'to_home'])
    df['ops_station_lat'] = df['ops station'].apply(lambda x: osm.fetch_coordinates(x)[0])
    df['ops_station_lon'] = df['ops station'].apply(lambda x: osm.fetch_coordinates(x)[1])
    df['start_lat'] = df['start'].apply(lambda x: osm.fetch_coordinates(x)[0])
    df['start_lon'] = df['start'].apply(lambda x: osm.fetch_coordinates(x)[1])
    df.drop(['ops station', 'start'], axis=1, inplace=True)
    osm.save_coordinates_dict()
    logging.info('finished coding stations')
    df.drop(df[df['update'] >= pd.to_datetime(mds.DefaultTrainingDateCut)].index, inplace=True)
    return df


def one_hot_encoding(df: pd.DataFrame, **kwargs) -> List | pd.DataFrame:
    encoder = kwargs.get('encoder', None)
    if encoder:
        encoded = encoder.transform(df[['o_road']])
        encoded_df = pd.DataFrame(encoded,
                                  columns=[f"o_road_{col}" for col in encoder.get_feature_names_out()])
        return pd.concat([df, encoded_df], axis=1)
    else:
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(df['o_road'].values.reshape(-1, 1))
        encoded_df = pd.DataFrame(encoded, columns=[f"o_road_{col}" for col in encoder.get_feature_names_out()])
        return [pd.concat([df, encoded_df], axis=1), encoder]


def preprocessing_update_trains(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(subset=['train_start', 'расстояние до Лены', 'op_station_index', 'ops station'], inplace=True)
    df['DLeft'] = df['расстояние до Лены'].astype(int)
    logging.info('starting coding stations')
    df['ops_station_lat'] = df['ops station'].apply(lambda x: osm.fetch_coordinates(x)[0])
    df['ops_station_lon'] = df['ops station'].apply(lambda x: osm.fetch_coordinates(x)[1])
    df['start_lat'] = df['start'].apply(lambda x: osm.fetch_coordinates(x)[0])
    df['start_lon'] = df['start'].apply(lambda x: osm.fetch_coordinates(x)[1])
    df.drop(['ops station', 'start'], axis=1, inplace=True)
    logging.info('finished coding stations')
    osm.save_coordinates_dict()
    df['o_road'] = df['ops road']
    df.dropna(subset=['расстояние до Лены'], inplace=True)
    return df


def smooth_data(data: pd.DataFrame, filter_type: str = 'none') -> pd.DataFrame:
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


def cross_validation_test(models: Dict, metrics_data: List) -> pd.DataFrame:
    scoring_metric = 'neg_mean_squared_error'
    num_folds = 5
    scores = {}
    for name, model in models.items():
        X_test, y_test = metrics_data[name]
        if name in mds.TF_models_list:
            logging.info('TensorFlow models, no scoring')
            scores[name] = [0, 0]
        elif name in mds.sklearn_list:
            logging.info(f'SKLearn model, scoring for model {name} started')
            cross_scores = cross_val_score(estimator=model, X=X_test, y=y_test, cv=num_folds, scoring=scoring_metric)
            scores[name] = [-cross_scores.mean(), cross_scores.std()]
            logging.info(f'scoring for model {name} finished')
    scores = pd.DataFrame(scores)
    scores.index = ['Mean', 'Std']
    return scores


def get_models_metrics(models: Dict, metrics_data: Dict) -> pd.DataFrame:
    models_metrics = {}
    for name, model in models.items():
        X_test, y_test = metrics_data[name]
        logging.info(f'predicting for model {name} started')
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        models_metrics[name] = [mae, mse, rmse]
        logging.info(f'predicting for model {name} finished')
    metrics = pd.DataFrame(models_metrics)
    metrics.index = ['MAE', 'MSE', 'RMSE']
    return metrics


def create_models(df: pd.DataFrame, columns_list: List) -> Dict:
    logging.info('Started preprocessing')
    preprocessed_df = preprocessing_trains(df)
    logging.info('Preprocessing done')
    trains = preprocessed_df.reset_index(drop=True)
    encoded_roads, road_encoder = one_hot_encoding(trains)
    logging.info('encoding roads done')
    y = smooth_data(trains['to_home'], mds.filter_type)
    logging.info('Filtering done')
    logging.info('Fitting models')
    fit_models = {}
    metrics_data = {}
    scalers = {}
    columns = {}
    for name, model in mds.models.items():
        logging.info(f'fitting model {name} started')
        scalers[name] = None
        if name in mds.TF_models_list:
            columns_list = mds.TF_DefaultColumns
        elif name in mds.sklearn_list:
            columns_list = mds.DefaultColumns
        columns_to_keep = [col for col in encoded_roads.columns.astype(str) if col.startswith('o_road_x0_')] + columns_list
        X = encoded_roads[columns_to_keep]
        columns[name] = columns_to_keep
        X_train, X_remains, y_train, y_remains = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_remains, y_remains, test_size=0.5, random_state=42)
        if name in mds.TF_models_list:
            X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
            X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
            y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
            X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
            y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
            model.compile(loss=mds.TF_loss[name], optimizer=mds.TF_optimizers[name], metrics=mds.TF_metrics)
            mds.TF_optimizers[name].build(model.trainable_variables)
            model.fit(X_train, y_train, epochs=mds.TF_number_of_epochs,
                      batch_size=mds.TF_batch_size, validation_data=(X_val, y_val))
            fit_models[name] = model
        elif name in mds.sklearn_list:
            fit_models[name] = model.fit(X_train, y_train)
        metrics_data[name] = [X_test, y_test]
        logging.info(f'fitting model {name} finished')
    logging.info('All models fitted')
    logging.info('Calculating metrics')
    metrics = get_models_metrics(fit_models, metrics_data)
    logging.info('Metrics calculated')
    logging.info(f"Metrics: \n{metrics.to_markdown()}")
    logging.info('Calculating scores')
    scores = cross_validation_test(fit_models, metrics_data)
    logging.info('Scores calculated')
    logging.info(f"Scores: \n{scores.to_markdown()}")
    return {'fit_models': fit_models,
            'encoders': {'road_encoder': road_encoder},
            'scalers': scalers, 'metrics': metrics, 'scores': scores, 'columns': columns}


def prediction(df: pd.DataFrame, models_dict: Dict) -> pd.DataFrame:
    logging.info('Started preprocessing')
    preprocessed_df = preprocessing_update_trains(df, models_dict['encoders'])
    encoded_roads = one_hot_encoding(preprocessed_df.reset_index(drop=True), encoder=models_dict['encoders']['road_encoder'])
    update_trains = encoded_roads
    logging.info('Preprocessing done')
    logging.info('Predicting')
    columns_to_keep = []
    for name, model in models_dict['models'].items():
        logging.info(f'predicting for model {name} started')
        update_X = update_trains[models_dict['columns'][name]]
        scaler = models_dict['scalers'][name]
        if scaler:
            update_X = scaler.transform(update_X)
        if name in mds.TF_models_list:
            update_X = tf.convert_to_tensor(update_X, dtype=tf.float32)
        update_Y = model.predict(update_X)
        logging.info(f'predicting for model {name} finished')
        duration = 'duration_' + name
        update_trains[duration] = pd.DataFrame(update_Y)
        expected_delivery = 'expected_delivery_' + name
        timedelta = pd.to_timedelta(update_trains[duration], unit='D')
        update_trains[expected_delivery] = pd.to_datetime(update_trains['update']) + timedelta
        update_trains[expected_delivery] = pd.to_datetime(update_trains[expected_delivery])
        columns_to_keep.append(duration)
        columns_to_keep.append(expected_delivery)
    logging.info('Predicting done')
    columns_to_keep.append('update')
    columns_to_keep.append('котлов')
    columns_to_keep.append('_num')
    return update_trains[columns_to_keep]


if __name__ == "__main__":
    pass

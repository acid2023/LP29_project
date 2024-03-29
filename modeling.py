import pandas as pd
import numpy as np
import datetime
import tensorflow as tf

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

import keras

import pickle
import logging
from typing import Dict, List, Union, Tuple
from tabulate import tabulate
import os

import modeling_settings as mds

import folders
from folders import folder_check
import osm

from preprocessing import PCA_training, PCA_encoding
from preprocessing import to_timestamp_days, to_datetime_days
from preprocessing import one_hot_encoder_training, one_hot_encoding
from preprocessing import initial_preprocessing, preprocessing_data
from preprocessing import get_no_leak, get_no_leak_stay


def save_models(
    models_dict: Dict[str, Union[
        Dict[str, object],
        Dict[str, Dict[str, OneHotEncoder]],
        Dict[str, StandardScaler],
        Dict[str, Tuple[float, float]],
        Dict[str, float], Dict[str, pd.DataFrame],
        Dict[str, List[str]]]]) -> None:
    path = folder_check(folders.models_folder)
    model_filename = f'{path}models_sklearn.pkl'
    sklearn_models = {}
    sklearn_list = []
    TF_models_list = []
    for model_name, model in models_dict['fit_models'].items():
        if model_name in mds.sklearn_list:
            sklearn_models[model_name] = model
            sklearn_list.append(model_name)
        else:
            model.save(f'{path}{model_name}.h5')
            TF_models_list.append(model_name)
    with open(model_filename, 'wb') as file:
        pickle.dump(sklearn_models, file)
    with open(f'{path}scalers.pkl', 'wb') as file:
        pickle.dump(models_dict['scalers'], file)
    with open(f'{path}TF_models_list.pkl', 'wb') as file:
        pickle.dump(TF_models_list, file)
    with open(f'{path}sklearn_list.pkl', 'wb') as file:
        pickle.dump(sklearn_list, file)
    with open(model_filename, 'wb') as file:
        pickle.dump(sklearn_models, file)
    with open(f'{path}models_metrics.pkl', 'wb') as file:
        pickle.dump(models_dict['metrics'], file)
    with open(f'{path}columns.pkl', 'wb') as file:
        pickle.dump(models_dict['columns'], file)
    with open(f'{path}encoders.pkl', 'wb') as file:
        pickle.dump(models_dict['encoders'], file)


def load_models() -> Dict[str, Union[Dict[str, object], Dict[str, StandardScaler],
                                     Dict[str, OneHotEncoder], Dict[str, List[str]]]]:
    path = folder_check(folders.models_folder)
    model_filename = f'{path}models_sklearn.pkl'
    with open(f'{path}columns.pkl', 'rb') as file:
        columns = pickle.load(file)
    with open(model_filename, 'rb') as file:
        models = pickle.load(file)
    with open(f'{path}TF_models_list.pkl', 'rb') as file:
        TF_models_list = pickle.load(file)
    for model_name in TF_models_list:
        with keras.utils.custom_object_scope({'MyPreprocessingLayer': mds.MyPreprocessingLayer}):
            models[model_name] = keras.models.load_model(f'{path}{model_name}.h5')
    scalers = pickle.load(open(f'{path}scalers.pkl', 'rb'))
    encoders = pickle.load(open(f'{path}encoders.pkl', 'rb'))
    return {'models': models, 'scalers': scalers, 'encoders': encoders, 'columns': columns}


def load_models_from_folder(folder_path):
    model_dict = {}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            # Load Keras models
            if filename.endswith('.h5') or filename.endswith('.keras'):
                model = keras.models.load_model(file_path)
                model_dict[filename] = model
            # Load non-Keras models
            elif filename.endswith('.pkl'):
                # Load your non-Keras models here and store them in the model_dict
                model = pickle.load(open(file_path, 'rb'))
                model_dict[filename] = model

    return model_dict


"""
def to_datetime_days(days_timestamp):
    return datetime.datetime.fromtimestamp(days_timestamp * 86400)


def to_timestamp_days(date):
    return int(datetime.datetime.timestamp(date) / 86400)
"""


def preprocessing_trains(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(subset=['DLeft', 'ops station', 'o_road', 'to_home'], inplace=True)
    df.reset_index(drop=True)
    df['in_train'] = df['in_train'].fillna(1)

    logging.error('starting coding stations')
    df['ops_station_lat'] = df['ops station'].apply(lambda x: float(osm.fetch_coordinates(x)[0]))
    df['ops_station_lon'] = df['ops station'].apply(lambda x: float(osm.fetch_coordinates(x)[1]))
    df.drop(['ops station'], axis=1, inplace=True)
    df.dropna(subset=['ops_station_lat', 'ops_station_lon'], inplace=True)
    df.reset_index(drop=True)
    osm.save_coordinates_dict()
    logging.error('finished coding stations')

    df.drop(df.loc[df['update'] >= pd.to_datetime(mds.DefaultTrainingDateCut)].index, inplace=True)
    df.reset_index(drop=True)
    logging.error('converting update times')
    df['update'] = pd.to_datetime(df['update']).apply(to_timestamp_days)
    logging.error('finished converting update times')
    return df.reset_index()


"""
def one_hot_encoder_training(df: pd.DataFrame | np.ndarray) -> OneHotEncoder:
    encoding_roads = df['o_road'].unique()
    encoding_roads_array = np.array(encoding_roads).astype(str)
    path = folder_check(folders.dict_folder)
    filename = f'{path}ops_road_list.pkl'
    if os.path.isfile(filename):
        loaded_roads = pickle.load(open(filename, 'rb'))
        loaded_roads_array = np.array(loaded_roads).astype(str)
        total_array = np.unique(np.union1d(loaded_roads_array, encoding_roads_array))
    else:
        total_array = encoding_roads_array
    data_to_train_encoder = pd.DataFrame(total_array, columns=['o_road'])
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(data_to_train_encoder['o_road'].values.reshape(-1, 1))
    return encoder



def one_hot_encoding(df: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    encoded = encoder.transform(df[['o_road']].values.reshape(-1, 1))
    encoded_df = pd.DataFrame(encoded, columns=[f"o_road_{col}" for col in encoder.get_feature_names_out()])
    return pd.concat([df, encoded_df], axis=1)
"""


def get_models_metrics(models: Dict[str, object],
                       metrics_data: Dict[str, Tuple[pd.DataFrame, pd.Series]]
                       ) -> pd.DataFrame:
    models_metrics = {}
    print(models)
    for name, model in models.items():
        X_test = metrics_data[name][0]
        y_test = metrics_data[name][1]
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


def create_models(
        df: pd.DataFrame, columns_list: List[str]
        ) -> Dict[str, Union[Dict[str, object],
                             Dict[str, Dict[str, OneHotEncoder]],
                             Dict[str, StandardScaler], Dict[str, Tuple[float, float]],
                             Dict[str, float], Dict[str, pd.DataFrame], Dict[str, List[str]]]]:
    logging.error('Started preprocessing')
    trains = preprocessing_trains(df)
    logging.error('Preprocessing done')

    road_encoder = one_hot_encoder_training(trains)
    encoded_roads = one_hot_encoding(trains, road_encoder)
    encoded_roads.reset_index()
    one_hot_encoded_columns = [col for col in encoded_roads.columns.astype(str) if col.startswith('o_road_x0_')]
    logging.error('encoding roads done')

    logging.error('Fitting models')
    fit_models = {}
    scalers = {}
    metrics_data = {}
    columns = {}
    keras_columns_list = mds.TF_DefaultColumns + one_hot_encoded_columns
    path = folder_check(folders.models_folder)

    # Create a MirroredStrategy
    strategy = tf.distribute.MirroredStrategy()

    # Create and compile the model within the strategy's scope
    with strategy.scope():

        # models = mds.declare_keras_models(mds.models, len(keras_columns_list), path)
        models = mds.declare_keras_models({}, len(keras_columns_list), path)
    TF_models_list = [model for model in models if model.startswith('TensorFlow')]

# saving preprocessed data for experiments
#    A = trains
#    b = encoded_roads['to_home']
#    A.to_pickle('/Users/sergeykuzmin/projects/project/LP29_project/inputs_no_encode.pkl')
#    b.to_pickle('/Users/sergeykuzmin/projects/project/LP29_project/target.pkl')
#    input()

    for name, model in models.items():
        logging.error(f'fitting model {name} started')
        if name in TF_models_list:
            columns_list = keras_columns_list
        elif name in mds.sklearn_list:
            columns_list = mds.DefaultColumns + one_hot_encoded_columns
        X = encoded_roads[columns_list]
        y = encoded_roads['to_home']
        columns[name] = columns_list
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        fit_models[name] = model.fit(X_train, y_train)
        metrics_data[name] = [X_test, y_test]
        logging.error(f'fitting model {name} finished')
    logging.error('All models fitted')
    metrics = {}
    scores = {}
    logging.error('Calculating metrics')
    metrics = get_models_metrics(fit_models, metrics_data)
    metrics = metrics.T
    table_data = metrics.reset_index().values.tolist()
    table_headers = [''] + list(metrics.columns)

    logging.error('Metrics calculated')
    table = tabulate(table_data, headers=table_headers, tablefmt='fancy_grid', showindex='never')
    logging.info(f"Metrics: \n{table}")

    scores = {}

    return {'fit_models': fit_models,
            'encoders': {'road_encoder': road_encoder},
            'scalers': scalers, 'metrics': metrics, 'scores': scores, 'columns': columns}


def create_2_way_models(df):
    update_cut = mds.DefaultTrainingDateCut
    one_hot_encoder = one_hot_encoder_training(df.copy())
    df_features = initial_preprocessing(df.copy())
    train, test = get_no_leak(df_features.copy(), update_cut)
    coded_train = preprocessing_data(train.copy())
    coded_test = preprocessing_data(test.copy())
    train_stay, test_stay = get_no_leak_stay(df.copy(), update_cut)
    coded_train_stay = preprocessing_data(train_stay.rename(columns={'at_client': 'target'}).copy())
    coded_test_stay = preprocessing_data(test_stay.rename(columns={'at_client': 'target'}).copy())
    PCA_training(coded_train.copy().drop(columns=['target', 'o_road']))
    PCA_train_encoded = PCA_encoding(coded_train.copy().drop(columns=['target', 'o_road']))
    PCA_train = pd.concat([PCA_train_encoded, coded_train[['target', 'o_road']]], axis=1)
    PCA_test_encoded = PCA_encoding(coded_test.copy().drop(columns=['target', 'o_road']))
    PCA_test = pd.concat([PCA_test_encoded, coded_test[['target', 'o_road']]], axis=1)
    OH_PCA_train = one_hot_encoding(PCA_train.copy(), one_hot_encoder)
    OH_PCA_test = one_hot_encoding(PCA_test.copy(), one_hot_encoder)
    OH_train = one_hot_encoding(coded_train.copy(), one_hot_encoder)
    OH_test = one_hot_encoding(coded_test.copy(), one_hot_encoder)

    no_PCA_num_features = len(OH_train.columns)
    PCA_num_features = len(OH_PCA_train.columns)
    stay_num_features = len(coded_train_stay.columns)
    if 'target' in coded_train_stay.columns:
        stay_num_features -= 1
        y_stay = coded_train_stay.pop('target')
    if 'target' in OH_train.columns:
        no_PCA_num_features -= 1
        y_no_PCA = OH_train.pop('target')
    if 'target' in OH_PCA_train.columns:
        PCA_num_features -= 1
        y_PCA = OH_PCA_train.pop('target')
    # Create a MirroredStrategy
    # strategy = tf.distribute.MirroredStrategy()

    path = folder_check(folders.models_folder)
    # Create and compile the model within the strategy's scope
    # with strategy.scope():

    if 'target' in coded_test_stay.columns:
        y_stay_test = coded_test_stay.pop('target')
    if 'target' in OH_PCA_test.columns:
        y_PCA_test = OH_PCA_test.pop('target')
    if 'target' in OH_test.columns:
        y_no_PCA_test = OH_test.pop('target')

    coded_val_stay, coded_test_stay, y_stay_val, y_stay_test = train_test_split(coded_test_stay, y_stay_test, train_size=0.5, random_state=42)
    OH_PCA_val, OH_PCA_test, y_PCA_val, y_PCA_test = train_test_split(OH_PCA_test, y_PCA_test, train_size=0.5, random_state=42)
    OH_val, OH_test, y_no_PCA_val, y_no_PCA_test = train_test_split(OH_test, y_no_PCA_test, train_size=0.5, random_state=42)

    fit_PCA_models = {}
    fit_no_PCA_models = {}
    fit_stay_models = {}
    PCA_metrics_data = {}
    no_PCA_metrics_data = {}
    stay_metrics_data = {}

    PCA_models = mds.declare_keras_models(mds.PCA_models, PCA_num_features, path)

    for name, model in PCA_models.items():
        logging.error(f'PCA - fitting model {name} started')
        if name in mds.PCA_models:
            fit_PCA_models[name] = PCA_models[name].fit(OH_PCA_train, y_PCA)
        else:
            fit_PCA_models[name] = PCA_models[name].fit(OH_PCA_train, y_PCA, validation_data=(OH_PCA_val, y_PCA_val))
        PCA_metrics_data[name] = [OH_PCA_test, y_PCA_test]
        logging.error(f'PCA - fitting model {name} finished')
    logging.error('All PCA models fitted')

    no_PCA_models = mds.declare_keras_models(mds.no_PCA_models, no_PCA_num_features, path)
    for name, model in no_PCA_models.items():
        logging.error(f'no PCA fitting model {name} started')
        if name in mds.models:
            fit_no_PCA_models[name] = model.fit(OH_train, y_no_PCA)
        else:
            fit_no_PCA_models[name] = model.fit(OH_train, y_no_PCA, validation_data=(OH_val, y_no_PCA_val))
        no_PCA_metrics_data[name] = [OH_test, y_no_PCA_test]
        logging.error(f'no PCA fitting model {name} finished')
    logging.error('All no_PCA models fitted')

    stay_models = mds.declare_keras_models(mds.stay_models, stay_num_features, path)
    for name, model in stay_models.items():
        logging.error(f'stay fitting model {name} started')
        if name in mds.models:
            fit_stay_models[name] = model.fit(coded_train_stay, y_stay)
        elif name.endswith('Flat'):
            continue
        else:
            fit_stay_models[name] = model.fit(coded_train_stay, y_stay, validation_data=(coded_val_stay, y_stay_val))
        stay_metrics_data[name] = [coded_test_stay, y_stay_test]
        logging.error(f'stay fitting model {name} finished')
    logging.error('All stay models fitted')

    metrics = get_models_metrics(fit_stay_models, stay_metrics_data).T
    table_data = metrics.reset_index().values.tolist()
    table_headers = [''] + list(metrics.columns)

    table = tabulate(table_data, headers=table_headers, tablefmt='fancy_grid', showindex='never')
    logging.info(f"Metrics for stay: \n{table}")

    metrics = get_models_metrics(fit_PCA_models, PCA_metrics_data).T
    table_data = metrics.reset_index().values.tolist()
    table_headers = [''] + list(metrics.columns)

    table = tabulate(table_data, headers=table_headers, tablefmt='fancy_grid', showindex='never')
    logging.info(f"Metrics for PCA: \n{table}")

    metrics = get_models_metrics(fit_no_PCA_models, no_PCA_metrics_data).T
    table_data = metrics.reset_index().values.tolist()
    table_headers = [''] + list(metrics.columns)

    table = tabulate(table_data, headers=table_headers, tablefmt='fancy_grid', showindex='never')
    logging.info(f"Metrics for no PCA: \n{table}")

    path = folder_check(folders.models_folder)
    for model_name, model in fit_PCA_models.items():
        if model_name in mds.models:
            with open(f'{path}PCA/{model_name}.pkl', 'wb') as file:
                pickle.dump(model, file)
        else:
            model.save(f'{path}PCA/{model_name}.h5')
    logging.error('All PCA models saved')

    for model_name, model in fit_no_PCA_models.items():
        if model_name in mds.models:
            with open(f'{path}no_PCA/{model_name}.pkl', 'wb') as file:
                pickle.dump(fit_no_PCA_models[model_name], file)
        else:
            model.save(f'{path}no_PCA/{model_name}.h5')
    logging.error('All no_PCA models saved')

    for model_name, model in stay_models.items():
        if model_name in mds.models:
            with open(f'{path}stay/{model_name}.pkl', 'wb') as file:
                pickle.dump(fit_stay_models[model_name], file)
        else:
            if model_name in fit_stay_models:
                fit_stay_models[model_name].save(f'{path}stay/{model_name}.h5')

    logging.error('All stay models saved')


def preprocessing_updates(input: pd.DataFrame) -> pd.DataFrame:
    df = input.copy()
    df.dropna(subset=['DLeft', 'ops station', 'o_road'], inplace=True)
    df.reset_index(drop=True)
    df['in_train'] = df['in_train'].fillna(1)

    logging.error('starting coding stations')
    # df = df[df['ops station'] != -904851]
    df['ops_station_lat'] = df['ops station'].apply(lambda x: float(osm.fetch_coordinates(x)[0]))
    df['ops_station_lon'] = df['ops station'].apply(lambda x: float(osm.fetch_coordinates(x)[1]))
    df.drop(['ops station'], axis=1, inplace=True)
    df.dropna(subset=['ops_station_lat', 'ops_station_lon'], inplace=True)
    df.reset_index(drop=True)
    osm.save_coordinates_dict()
    logging.info('finished coding stations')

    df.drop(df.loc[df['update'] < pd.to_datetime(mds.DefaultTrainingDateCut)].index, inplace=True)
    df.reset_index(drop=True)

    logging.info('converting update times')
    df['keep_update'] = df['update']
    df['update'] = pd.to_datetime(df['update']).apply(to_timestamp_days)
    logging.info('finished converting update times')

    return df.reset_index()


def prediction(df: pd.DataFrame) -> pd.DataFrame:
    logging.error('Loading models')
    models_dict = load_models()
    logging.error('Models loaded')

    logging.error('Started preprocessing')
    preprocessed_df = preprocessing_updates(df)
    road_encoder = models_dict['encoders']['road_encoder']
    update_trains = one_hot_encoding(preprocessed_df, road_encoder)
    logging.error('Preprocessing done')

    logging.error('Predicting')
    columns_to_keep = []
    for name, model in models_dict['models'].items():
        logging.info(f'predicting for model {name} started')
        update_X = update_trains[models_dict['columns'][name]]
        update_X.dropna()
        update_X.reset_index(drop=True)
        update_Y = model.predict(update_X)
        logging.info(f'predicting for model {name} finished')
        logging.info('setting/coverting times')
        duration = 'duration_' + name
        update_trains[duration] = pd.DataFrame(update_Y)
        expected_delivery = 'expected_delivery_' + name
        timedelta = pd.to_timedelta(update_trains[duration], unit='D')
        update_trains[expected_delivery] = update_trains['keep_update'] + timedelta
        update_trains[expected_delivery] = pd.to_datetime(update_trains[expected_delivery])
        logging.info('setting/coverting times done')
        columns_to_keep.append(duration)
        columns_to_keep.append(expected_delivery)
    logging.error('Predicting done')
    update_trains['update'] = update_trains['keep_update']
    columns_to_keep.append('update')
    columns_to_keep.append('котлов')
    columns_to_keep.append('_num')
    return update_trains[columns_to_keep]


def cross_validation_test(models: Dict[str, object],
                          metrics_data: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> pd.DataFrame:
    def cross_validate(
            model: object, name: str, X: pd.DataFrame, y: pd.Series,
            n_splits: int = 5
            ) -> Tuple[float, float]:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_index, test_index in kf.split(X):
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            scores.append(score)
        return -np.mean(scores), np.std(scores)

    num_folds = 5
    scores = {}
    for name, model in models.items():
        X_test, y_test = metrics_data[name]
        logging.info(f'scoring for model {name} started')
        scores[name] = cross_validate(model, name, X_test, y_test, n_splits=num_folds)
        logging.info(f'scoring for model {name} finished')
    scores = pd.DataFrame.from_dict(scores, orient='columns')
    scores.index = ['Mean', 'Std']
    return scores


def validate_models(df: pd.DataFrame) -> pd.DataFrame:
    logging.error('Loading models')
    models_dict = load_models()
    logging.error('Models loaded')

    logging.error('Started preprocessing')
    trains = preprocessing_trains(df)
    trains.reset_index()
    logging.error('Preprocessing done')

    road_encoder = one_hot_encoder_training(trains)
    encoded_roads = one_hot_encoding(trains, road_encoder)
    encoded_roads.reset_index()
    logging.error('encoding roads done')

    metrics_data = {}
    scores = {}

    logging.error('preparing metrics for validation')
    for name, model in models_dict['models'].items():
        columns_list = models_dict['columns'][name]
        encoded_roads.reset_index()
        X = encoded_roads[columns_list]
        y = encoded_roads['to_home']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        metrics_data[name] = [X_test, y_test]
    logging.error('finished metrics for validation')

    logging.error('starting cross validation')
    scores = cross_validation_test(models_dict['models'], metrics_data)
    logging.error('finished cross validation')

    path = folder_check(folders.models_folder)
    with open(f'{path}cross_validation_scores.pkl', 'wb') as file:
        pickle.dump(scores, file)

    table = tabulate(scores.T, headers='keys', tablefmt='pipe')

    logging.error(f'resulting scores:\n{table}')


def preprocessing_updates_post_modeling(input: pd.DataFrame) -> pd.DataFrame:
    df = input.copy()
    df.dropna(subset=['DLeft', 'ops station', 'o_road', 'to_home'], inplace=True)
    df.reset_index(drop=True)
    df['in_train'] = df['in_train'].fillna(1)

    logging.error('starting coding stations')
    # df = df[df['ops station'] != -904851]
    df['ops_station_lat'] = df['ops station'].apply(lambda x: float(osm.fetch_coordinates(x)[0]))
    df['ops_station_lon'] = df['ops station'].apply(lambda x: float(osm.fetch_coordinates(x)[1]))
    df.drop(['ops station'], axis=1, inplace=True)
    df.dropna(subset=['ops_station_lat', 'ops_station_lon'], inplace=True)
    df.reset_index(drop=True)
    osm.save_coordinates_dict()
    logging.info('finished coding stations')

    df.drop(df.loc[df['update'] < pd.to_datetime(mds.DefaultTrainingDateCut)].index, inplace=True)
    df.reset_index(drop=True)

    logging.info('converting update times')
    df['keep_update'] = df['update']
    df['update'] = pd.to_datetime(df['update']).apply(to_timestamp_days)
    logging.info('finished converting update times')

    return df.reset_index()


def validating_on_post_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.error('Loading models')
    models_dict = load_models()
    logging.error('Models loaded')

    logging.error('Started preprocessing')
    trains = preprocessing_updates_post_modeling(df)
    trains.reset_index()
    logging.error('Preprocessing done')

    road_encoder = one_hot_encoder_training(trains)
    encoded_roads = one_hot_encoding(trains, road_encoder)
    encoded_roads.reset_index()
    logging.error('encoding roads done')

    encoded_roads.dropna(subset=['to_home'])
    encoded_roads.reset_index(drop=True)

    data_dict = {str(date.date()): grp for date, grp in encoded_roads.groupby('keep_update')}
    logging.error('starting validation on post modeling data')
    metrics_data = {}
    metrics = {}

    for date, grp in data_dict.items():
        metrics_data[date] = {}
        logging.error(f'validation on date {date} started')
        for name in models_dict['models'].keys():
            columns_list = models_dict['columns'][name]
            X = grp[columns_list]
            y = grp['to_home']
            metrics_data[date][name] = [X, y]
        metrics[date] = get_models_metrics(models_dict['models'], metrics_data[date])
        logging.error(f'validation on date {date} finished')
    logging.error('finished validation on post modeling data')

    metrics_by_model = {model: {date: metrics[date][model] for date in metrics}
                        for model in metrics[list(metrics.keys())[0]]}

    mean_metrics_by_model = {}
    for model in metrics_by_model:
        mean_metrics = []
        for metric in range(3):
            metric_values = [metrics_by_model[model][date][metric] for date in metrics_by_model[model]]
            mean_metrics.extend([np.mean(metric_values), np.std(metric_values)])

        for date in metrics_by_model[model]:
            mean_mae = np.mean([metrics_by_model[model][date][2]])
            mean_metrics.append(mean_mae)

        mean_metrics_by_model[model] = mean_metrics

    metrics_df = pd.DataFrame.from_dict(mean_metrics_by_model,
                                        orient='index',
                                        columns=['mean_mse', 'std_mse',
                                                 'mean_mae', 'std_mae',
                                                 'mean_rmse', 'std_rmse'] + list(metrics_by_model[model].keys()))

    metrics_df = metrics_df.set_index(pd.Index(list(mean_metrics_by_model.keys()), name='Model'))
    metrics_df = metrics_df.sort_values(by=['mean_mae'], ascending=True)
    metrics_df.index.name = 'Model'
    table_data = metrics_df.reset_index().values.tolist()
    table_headers = [''] + list(metrics_df.columns)
    table = tabulate(table_data, headers=table_headers, tablefmt='fancy_grid', showindex='never')
    logging.error(f"resulting metrics:\n{table}")

    path = folder_check(folders.models_folder)
    with open(f'{path}metrics_post_modeling.pkl', 'wb') as file:
        pickle.dump(metrics_df, file)


if __name__ == "__main__":
    pass

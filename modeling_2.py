import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold


import keras

import pickle
import logging
from tabulate import tabulate
from typing import Dict, List, Union, Tuple
import os
import re

import modeling_settings as mds

import folders
from folders import folder_check
import osm

from preprocessing import PCA_training, PCA_encoding
from preprocessing import to_timestamp_days
from preprocessing import one_hot_encoder_training, one_hot_encoding, get_one_hot_encoder
from preprocessing import initial_preprocessing, preprocessing_data
from preprocessing import get_no_leak, get_no_leak_stay, get_no_leak_predict


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


def create_2_way_models(df):
    update_cut = mds.DefaultTrainingDateCut
    one_hot_encoder = one_hot_encoder_training(df.copy())
    df_features = initial_preprocessing(df.copy())
    df_features.drop(columns=['_num', 'котлов'], axis=1, inplace=True)
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


def prediction_2_way(df: pd.DataFrame) -> pd.DataFrame:
    logging.error('Loading models')
    PCA_models = load_models_from_folder('/models/PCA/')
    no_PCA_models = load_models_from_folder('/models/no_PCA/')
    logging.error('Models loaded')

    logging.error('Started preprocessing')

    update_cut = mds.DefaultTrainingDateCut
    df_features = initial_preprocessing(df.copy())
    wagon = df_features['_num', 'котлов', 'dest']
    df_features.drop(columns=['_num', 'котлов', 'dest'], axis=1, inplace=True)
    update = get_no_leak_predict(df_features.copy(), update_cut)
    update['keep_update'] = update['update']
    update = preprocessing_data(update.copy())
    PCA_update_encoded = PCA_encoding(update.copy().drop(columns=['target', 'o_road', 'keep_update']))
    PCA_update = pd.concat([PCA_update_encoded, update[['o_road', 'keep_update']]], axis=1)
    one_hot_encoder = get_one_hot_encoder(df.copy())

    OH_PCA_update = one_hot_encoding(PCA_update.copy(), one_hot_encoder)
    OH_PCA_update = pd.concat([OH_PCA_update, wagon], axis=1)
    OH_PCA_update = df_features.merge(OH_PCA_update, left_index=True, right_index=True, how='left')

    OH_update = one_hot_encoding(update.copy(), one_hot_encoder)
    OH_update = pd.concat([OH_update, wagon], axis=1)
    OH_update = df_features.merge(OH_update, left_index=True, right_index=True, how='left')

    logging.error('Preprocessing done')

    columns_to_remove = ['keep_update', '_num', 'котлов']

    update_X_PCA = OH_PCA_update.copy()
    update_X_PCA.drop(columns=columns_to_remove, axis=1, inplace=True)

    update_X_no_PCA = OH_update.copy()
    update_X_no_PCA.drop(columns=columns_to_remove, axis=1, inplace=True)

    for name, model in PCA_models.items():
        logging.info(f'predicting for PCA model {name} started')
        update_Y = model.predict(update_X_PCA)
        logging.info(f'predicting for model {name} finished')
        logging.info('setting/coverting times')
        duration = 'duration_' + name
        expected_delivery = 'expected_delivery_' + name
        OH_PCA_update[duration] = pd.DataFrame(update_Y)
        timedelta = pd.to_timedelta(OH_PCA_update[duration], unit='D')
        OH_PCA_update[expected_delivery] = OH_PCA_update['keep_update'] + timedelta
        OH_PCA_update[expected_delivery] = pd.to_datetime(OH_PCA_update[expected_delivery])
        logging.info('setting/coverting times done')
    logging.error('Predicting done')
    OH_PCA_update['update'] = OH_PCA_update['keep_update']

    for name, model in no_PCA_models.items():
        logging.info(f'predicting for no PCA model {name} started')
        update_Y = model.predict(update_X_no_PCA)
        logging.info(f'predicting for model {name} finished')
        logging.info('setting/coverting times')
        duration = 'duration_' + name
        expected_delivery = 'expected_delivery_' + name
        OH_update[duration] = pd.DataFrame(update_Y)
        timedelta = pd.to_timedelta(OH_update[duration], unit='D')
        OH_update[expected_delivery] = OH_update['keep_update'] + timedelta
        OH_update[expected_delivery] = pd.to_datetime(OH_update[expected_delivery])
        logging.info('setting/coverting times done')
    logging.error('Predicting done')
    OH_update['update'] = OH_update['keep_update']

    prefixes_to_keep = ['duration_', 'expected_delivery_', 'update', '_num', 'котлов']
    pattern = r'^(' + '|'.join(prefixes_to_keep) + ')'

    # Use the filter function to keep the matching columns
    filtered_OH = OH_update.filter(regex=pattern)
    filtered_OH_PCA = OH_PCA_update.filter(regex=pattern)

    update = pd.concat([filtered_OH, filtered_OH_PCA], ignore_index=True)

    return update


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
    models_dict = {}#load_models()
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

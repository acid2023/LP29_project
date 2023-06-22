import pandas as pd
import numpy as np
import datetime

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scipy.signal import savgol_filter, filtfilt, butter

import keras
import keras_tuner as kt
import tensorflow as tf

import pickle
import logging
from typing import Dict, List, Union, Tuple
from tabulate import tabulate
import os

import modeling_settings as mds
import folders
from folders import folder_check
import osm


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
        models[model_name] = keras.models.load_model(f'{path}{model_name}.h5')
    scalers = pickle.load(open(f'{path}scalers.pkl', 'rb'))
    encoders = pickle.load(open(f'{path}encoders.pkl', 'rb'))
    return {'models': models, 'scalers': scalers, 'encoders': encoders, 'columns': columns}


def to_datetime_days(days_timestamp):
    return datetime.datetime.fromtimestamp(days_timestamp * 86400)


def to_timestamp_days(date):
    return int(datetime.datetime.timestamp(date) / 86400)


def preprocessing_trains(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(subset=['DLeft', 'ops station', 'o_road', 'to_home'], inplace=True)
    df.reset_index(drop=True)
    df['in_train'] = df['in_train'].fillna(1)

    logging.error('starting coding stations')
    df = df[df['ops station'] != -904851]
    df['ops_station_lat'] = df['ops station'].apply(lambda x: osm.fetch_coordinates(x)[0])
    df['ops_station_lon'] = df['ops station'].apply(lambda x: osm.fetch_coordinates(x)[1])
    df.drop(['ops station'], axis=1, inplace=True)
    df.dropna(subset=['ops_station_lat', 'ops_station_lon'], inplace=True)
    df.reset_index(drop=True)
    osm.save_coordinates_dict()
    logging.error('finished coding stations')

    df.drop(df.loc[df['update'] >= pd.to_datetime(mds.DefaultTrainingDateCut)].index, inplace=True)
    df.reset_index(drop=True)
    logging.error('coverting update times')
    df['update'] = pd.to_datetime(df['update']).apply(to_timestamp_days)
    logging.error('finished converting update times')
    return df.reset_index()


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


def get_models_metrics(models: Dict[str, object],
                       metrics_data: Dict[str, Union[Tuple[pd.DataFrame, pd.Series], Tuple[tf.Tensor, tf.Tensor]]]
                       ) -> pd.DataFrame:
    models_metrics = {}
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

    # encoded_roads['to_home'] = smooth_data(encoded_roads['to_home'], mds.filter_type)
    # logging.info('Filtering done')

    logging.error('Fitting models')
    fit_models = {}
    scalers = {}
    metrics_data = {}
    columns = {}
    keras_columns_list = columns_list = mds.DefaultColumns + one_hot_encoded_columns
    path = folder_check(folders.models_folder)

    models = mds.declare_keras_models(mds.models, len(keras_columns_list), path)
    TF_models_list = [model for model in models if model.startswith('TensorFlow')]

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
        if name in TF_models_list:
            logging.error(f'Model {name}: epoch with min val_los - {model.get_min_val_loss_epoch()}')
        metrics_data[name] = [X_test, y_test]
        logging.error(f'fitting model {name} finished')
    logging.error('All models fitted')

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


def preprocessing_updates(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(subset=['DLeft', 'ops station', 'o_road', 'to_home'], inplace=True)
    df.reset_index(drop=True)
    df['in_train'] = df['in_train'].fillna(1)

    logging.error('starting coding stations')
    df = df[df['ops station'] != -904851]
    df['ops_station_lat'] = df['ops station'].apply(lambda x: osm.fetch_coordinates(x)[0])
    df['ops_station_lon'] = df['ops station'].apply(lambda x: osm.fetch_coordinates(x)[1])
    df.drop(['ops station'], axis=1, inplace=True)
    df.dropna(subset=['ops_station_lat', 'ops_station_lon'], inplace=True)
    df.reset_index(drop=True)
    osm.save_coordinates_dict()
    logging.info('finished coding stations')

    df.drop(df.loc[df['update'] < pd.to_datetime(mds.DefaultTrainingDateCut)].index, inplace=True)
    df.reset_index(drop=True)

    logging.info('coverting update times')
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
        if name.startswith('TensorFlow'):
            update_X = tf.convert_to_tensor(update_X, dtype=tf.float32)
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
            if name.startswith('TensorFlow'):
                X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            scores.append(score)
        return np.mean(scores), np.std(scores)

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

    encoded_roads['to_home'] = smooth_data(encoded_roads['to_home'], mds.filter_type)
    logging.info('Filtering done')

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


def validating_on_post_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.error('Loading models')
    models_dict = load_models()
    logging.error('Models loaded')

    logging.error('Started preprocessing')
    trains = preprocessing_updates(df)
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
            if name.startswith('TensorFlow'):
                X = tf.convert_to_tensor(X, dtype=tf.float32)
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
        mean_metrics_by_model[model] = mean_metrics

    metrics_df = pd.DataFrame.from_dict(mean_metrics_by_model,
                                        orient='index',
                                        columns=['mean_mse', 'std_mse',
                                                 'mean_mae', 'std_mae',
                                                 'mean_rmse', 'std_rmse'])

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


def KeraTune():

    path = folder_check(folders.models_folder)
    logging.error('starting tuning')

    with open(f'{path}X.pkl', 'rb') as file:
        X = pickle.load(file)
    with open(f'{path}y.pkl', 'rb') as file:
        y = pickle.load(file)
    with open(f'{path}keras_list.pkl', 'rb') as file:
        keras_list = pickle.load(file)
    global num_features
    num_features = len(keras_list)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    global input_shape
    input_shape = (num_features,)

    def model_builder(hp):
        global input_shape
        model = keras.Sequential()
        num_layers = hp.Int('num_layers', min_value=5, max_value=8, step=1)
        for i in range(num_layers):
            hp_units = hp.Int('units_{}'.format(i), min_value=96, max_value=512, step=32)
            hp_activation = hp.Choice('activation_{}'.format(i), values=['elu', 'relu', 'selu', 'softplus'])
            if i == 0:
                model.add(keras.layers.Dense(units=hp_units, activation=hp_activation, input_shape=input_shape))
            else:
                model.add(keras.layers.Dense(units=hp_units, activation=hp_activation))
        model.add(keras.layers.Dense(1))
        hp_optimizer = hp.Choice('optimizer', values=['adam', 'nadam', 'rmsprop'])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        if hp_optimizer == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
        elif hp_optimizer == 'nadam':
            optimizer = keras.optimizers.Nadam(learning_rate=hp_learning_rate)
        else:
            optimizer = keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])
        return model

    def conv_model(hp):
        model = keras.Sequential()
        global num_features
        input_shape = (num_features,)
        # Add a dense layer with a hyperparameter for the activation function
        hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
        model.add(keras.layers.Dense(units=34, activation=hp_activation, input_shape=input_shape))

        # Reshape the output of the dense layer
        model.add(keras.layers.Reshape((num_features, 1)))

        # Add attention layers as hyperparameters
        use_attention = True
        if use_attention:
            hp_attention_units = hp.Int('attention_units', min_value=16, max_value=64, step=16)
            hp_attention_activation = hp.Choice('attention_activation', values=['sigmoid', 'softmax'])
            query = keras.layers.Input(shape=(None, num_features))
            value = keras.layers.Input(shape=(None, num_features))
            attention = keras.layers.Attention()
            query_transformed = keras.layers.Dense(hp_attention_units, activation=hp_attention_activation)(query)
            value_transformed = keras.layers.Dense(hp_attention_units, activation=hp_attention_activation)(value)
            attended_values = attention([query_transformed, value_transformed])
            attended_values = keras.layers.Input(tensor=attended_values, name='attended_values')
            model.add(attended_values)

        # Add convolutional layers with hyperparameters for filters, kernel size, and activation
        filters = hp.Int('filters', min_value=2, max_value=32, step=2)

        conv_activation = hp.Choice('conv_activation', values=['relu', 'tanh'])

        model.add(keras.layers.Conv1D(filters=filters, kernel_size=3, activation=conv_activation))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling1D(pool_size=2))

        model.add(keras.layers.Conv1D(filters=filters*2, kernel_size=5, activation=conv_activation))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling1D(pool_size=3))

        model.add(keras.layers.Conv1D(filters=filters*4, kernel_size=3, activation=conv_activation))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling1D(pool_size=2))

        # Add a flatten layer and a dense layer with a hyperparameter for the activation function
        dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        dense_activation = hp.Choice('dense_activation', values=['relu', 'tanh'])

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=dense_units, activation=dense_activation))
        model.add(keras.layers.Dropout(dropout_rate))

        # Add an output layer
        model.add(keras.layers.Dense(1))

        # Compile the model with hyperparameters for the optimizer and learning rate
        hp_optimizer = hp.Choice('optimizer', values=['adam', 'nadam', 'rmsprop'])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        if hp_optimizer == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
        elif hp_optimizer == 'nadam':
            optimizer = keras.optimizers.Nadam(learning_rate=hp_learning_rate)
        else:
            optimizer = keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

        return model

    class LoggingCallback(tf.keras.callbacks.Callback):
        def __init__(self, log_dir):
            super(LoggingCallback, self).__init__()
            self.log_dir = log_dir
            self.log_file = os.path.join(log_dir, 'tuner.log')

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Epoch {epoch} start time: {self.epoch_start_time}\n")

        def on_epoch_end(self, epoch, logs=None):
            epoch_end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"Epoch {epoch} end time: {epoch_end_time}\n")

            logging.info("Epoch summary:")
            for metric, value in logs.items():
                logging.info(f"{metric}: {value:.4f}")

    log_callback = LoggingCallback(folder_check(folders.logs_folder))
    tuner = kt.Hyperband(
     #   cmodel_builder, objective='val_loss',  continue later
                         conv_model, objective='val_loss',
                  #       directory='my_dir', project_name='regression_1', contimue later
                         directory='my_dir', project_name='regression_2',
                         overwrite=False)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[log_callback, stop_early])

    best_model = tuner.get_best_models(num_models=3)[0]

    test_loss, test_mae = best_model.evaluate(X_val, y_val)
    print('Test MAE:', test_mae)


if __name__ == "__main__":
    pass

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, Lars, LassoLars, OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge, ARDRegression, PassiveAggressiveRegressor, RANSACRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras import layers
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf

import pandas as pd

from typing import Dict
import logging

import time

import folders
from folders import folder_check


def log_settings() -> None:
    unix_time = int(time.time())
    with open(f'{__file__}', 'r') as file:
        module_text = file.read()
    path = folder_check(folders.logs_folder)
    filename = f'{path}/{unix_time}modeling_settings.ini'
    with open(filename, 'w') as f:
        f.write(str(module_text))


DefaultTrainingDateCut = '2023-05-15'
filter_types = ['savgol', 'butter', 'none']
filter_type = filter_types[2]

DefaultColumns = ['DLeft', 'ops_station_lat', 'ops_station_lon', 'update', 'in_train']

TF_DefaultColumns = ['DLeft', 'ops_station_lat', 'ops_station_lon', 'update', 'in_train']
TF_number_of_epochs = 100
TF_batch_size = 32
TF_neurons = 512
TF_learning_rate = 0.001
TF_input_shape = (None, )


class CustomKerasRegressor(KerasRegressor, BaseEstimator, RegressorMixin):
    def __init__(self, name: str, filepath: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_ = None
        self.name_ = type(self).__name__ + '_' + name
        self.filepath_ = filepath

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> keras.Sequential:

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

        self.epoch_metrics_ = {}
        log_metrics = keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch,
            logs: logging.info(f'Epoch {epoch}: {logs}'))
        update_metrics = keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch,
            logs: self.epoch_metrics_.update({epoch: {'loss': logs['loss'], 'val_loss': logs['val_loss']}}))
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tboard/logs")
        kwargs['callbacks'] = []
        kwargs['callbacks'].append(log_metrics)
        kwargs['callbacks'].append(update_metrics)
        kwargs['callbacks'].append(early_stop)
        kwargs['callbacks'].append(tensorboard_callback)

        history = super().fit(X_train, y_train, validation_data=(X_val, y_val), **kwargs)
        self.model_ = history.model
        return self.model_

    def get_min_val_loss_epoch(self) -> int:
        min_val_loss_epoch = min(self.epoch_metrics_, key=lambda x: self.epoch_metrics_[x]['val_loss'])
        return min_val_loss_epoch


def declare_keras_models(models_dict: Dict[str, object], num_features: int, filepath: str) -> Dict[str, object]:
    def TensorFlow_Relu_Elu_Selu_Nadam() -> keras.Sequential:
        model = keras.Sequential([layers.Dense(TF_neurons, activation='relu', input_shape=(None, num_features)),
                                  layers.Dense(TF_neurons, activation='elu'),
                                  layers.Dense(TF_neurons, activation='selu'),
                                  layers.Dense(1)])
        optimizer = keras.optimizers.Nadam(learning_rate=TF_learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def TensorFlow_Softplus_Nadam() -> keras.Sequential:
        model = keras.Sequential([layers.Dense(TF_neurons, activation='softplus', input_shape=(None, num_features)),
                                  layers.Dense(TF_neurons, activation='softplus'),
                                  layers.Dense(TF_neurons, activation='softplus'),
                                  layers.Dense(1)])
        optimizer = keras.optimizers.Nadam(learning_rate=TF_learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def TensorFlow_Synthetic() -> keras.Sequential:
        model = keras.Sequential([layers.Dense(TF_neurons, activation='relu', input_shape=(None, num_features)),
                                  layers.Dense(TF_neurons * 2, activation='softplus'),
                                  layers.Dense(TF_neurons * 2, activation='relu'),
                                  layers.Dense(1)])
        optimizer = keras.optimizers.Adam(learning_rate=TF_learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def TensorFlow_KeraTune_1() -> keras.Sequential:
        model = keras.Sequential([layers.Dense(192, activation='selu', input_shape=(None, num_features)),
                                  layers.Dense(160, activation='selu'),
                                  layers.Dense(224, activation='relu'),
                                  layers.Dense(96, activation='relu'),
                                  layers.Dense(96, activation='softplus'),
                                  layers.Dense(1)])
        optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def TensorFlow_KeraTune_2() -> keras.Sequential:
        model = keras.Sequential([layers.Dense(96, activation='relu', input_shape=(None, num_features)),
                                  layers.Dense(448, activation='elu'),
                                  layers.Dense(480, activation='elu'),
                                  layers.Dense(288, activation='relu'),
                                  layers.Dense(128, activation='elu'),
                                  layers.Dense(128, activation='softplus'),
                                  layers.Dense(352, activation='relu'),
                                  layers.Dense(256, activation='softplus'),
                                  layers.Dense(1)])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def TensorFlow_KeraTune_Conv_1() -> keras.Sequential:
        model = keras.Sequential()
        input_shape = (num_features,)
        model.add(keras.layers.Dense(units=34, activation='relu', input_shape=input_shape))
        model.add(keras.layers.Reshape((num_features, 1)))

        query = keras.layers.Input(shape=(None, num_features))
        value = keras.layers.Input(shape=(None, num_features))
        attention = keras.layers.Attention()
        query_transformed = keras.layers.Dense(64, activation='sigmoid')(query)
        value_transformed = keras.layers.Dense(64, activation='sigmoid')(value)
        attended_values = attention([query_transformed, value_transformed])
        attended_values = keras.layers.Input(tensor=attended_values, name='attended_values')
        model.add(attended_values)

        # Add convolutional layers with hyperparameters for filters, kernel size, and activation

        model.add(keras.layers.Conv1D(filters=30, kernel_size=3, activation='tanh'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling1D(pool_size=2))

        model.add(keras.layers.Conv1D(filters=60, kernel_size=5, activation='tanh'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling1D(pool_size=3))

        model.add(keras.layers.Conv1D(filters=120, kernel_size=3, activation='tanh'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling1D(pool_size=2))

        # Add a flatten layer and a dense layer with a hyperparameter for the activation function
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=96, activation='relu'))
        model.add(keras.layers.Dropout(0))

        # Add an output layer
        model.add(keras.layers.Dense(1))

        # Compile the model with hyperparameters for the optimizer and learning rate

        optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

        return model

    keras_models = {
        'TensorFlow_Relu_Elu_Selu_Nadam':
        CustomKerasRegressor('TensorFlow_Relu_Elu_Selu_Nadam', filepath, build_fn=TensorFlow_Relu_Elu_Selu_Nadam,
                             batch_size=TF_batch_size, epochs=TF_number_of_epochs),
        'TensorFlow_Softplus_Nadam':
        CustomKerasRegressor('TensorFlow_Softplus_Nadam', filepath, build_fn=TensorFlow_Softplus_Nadam,
                             batch_size=TF_batch_size, epochs=TF_number_of_epochs),
        'TensorFlow_Synthetic':
        CustomKerasRegressor('TensorFlow_Synthetic', filepath, build_fn=TensorFlow_Synthetic,
                             batch_size=TF_batch_size, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_1':
        CustomKerasRegressor('TensorFlow_KeraTune_1', filepath, build_fn=TensorFlow_KeraTune_1,
                             batch_size=TF_batch_size, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_2':
        CustomKerasRegressor('TensorFlow_KeraTune_2', filepath, build_fn=TensorFlow_KeraTune_2,
                             batch_size=TF_batch_size, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_Conv_1':
        CustomKerasRegressor('TensorFlow_KeraTune_Conv_1', filepath, build_fn=TensorFlow_KeraTune_Conv_1,
                             batch_size=TF_batch_size, epochs=TF_number_of_epochs)
                             }

    for model in keras_models:
        models_dict[model] = keras_models[model]

    return models_dict


models = {'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42, max_depth=1000),
          'DecisionTree': DecisionTreeRegressor(max_depth=1000, random_state=42),
          'KNeighbors': KNeighborsRegressor(n_neighbors=5),
          'ExtraTrees': ExtraTreesRegressor(random_state=42, n_estimators=500, max_depth=1000),
          'GradientBoosting': GradientBoostingRegressor(n_estimators=500, learning_rate=0.25, random_state=42),
          'Ridge': Ridge(alpha=1.0),
          'Lasso': Lasso(alpha=1.0),
          'Lars': Lars(n_nonzero_coefs=10),
          'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
          'BayesianRidge': BayesianRidge(),
          'ARDRegression': ARDRegression(),
          'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
          'RANSACRegressor': RANSACRegressor(),
          'ElasticNet': ElasticNet(),
          'LassoLars': LassoLars(),
          'AdaBoost': AdaBoostRegressor(random_state=42, n_estimators=500)}


sklearn_list = list(models.keys())

log_settings()

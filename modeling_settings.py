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


DefaultTrainingDateCut = '2023-05-30'
filter_types = ['savgol', 'butter', 'none']
filter_type = filter_types[2]

DefaultColumns = ['DLeft', 'ops_station_lat', 'ops_station_lon', 'update', 'in_train']

TF_DefaultColumns = ['DLeft', 'ops_station_lat', 'ops_station_lon', 'update', 'in_train']
TF_number_of_epochs = 100
TF_batch_size = 256
TF_neurons = 512
TF_learning_rate = 0.001
TF_input_shape = (None, )


class CustomKerasRegressor(KerasRegressor, BaseEstimator, RegressorMixin):
    def __init__(self, name, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_ = None
        self.name_ = name

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> keras.Sequential:

        if not self.name_.endswith('MultiInput'):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
            X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
            y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
            kwargs['validation_data'] = (X_val, y_val)
        self.epoch_metrics_ = {}
        log_metrics = keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch,
            logs: logging.info(f'Epoch {epoch}: {logs}'))
        update_metrics = keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch,
            logs: self.epoch_metrics_.update({epoch: {'loss': logs['loss'], 'val_loss': logs['val_loss']}}))
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"./tboard/logs2/{self.name_}")
        if kwargs.get('callbacks', None) is None:
            kwargs['callbacks'] = []
        # kwargs['callbacks'].append(log_metrics)
        kwargs['callbacks'].append(update_metrics)
        kwargs['callbacks'].append(early_stop)
        kwargs['callbacks'].append(tensorboard_callback)

        history = super().fit(X, y, **kwargs)

        self.model_ = history.model
        return self.model_

    def get_min_val_loss_epoch(self) -> int:
        min_val_loss_epoch = min(self.epoch_metrics_, key=lambda x: self.epoch_metrics_[x]['val_loss'])
        return min_val_loss_epoch


class MyPreprocessingLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyPreprocessingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MyPreprocessingLayer, self).build(input_shape)

    def call(self, inputs):
        def apply_make_matrix(row):
            features = row[:5]
            features = tf.reshape(features, (-1, 1))
            location = row[5:]
            location = tf.reshape(location, (1, -1))  # Reshape location tensor
            result = tf.multiply(features, location)
            return result

        processed_input = tf.map_fn(apply_make_matrix, inputs)
        return processed_input


def declare_keras_models(models_dict: Dict[str, object], num_features: int, filepath: str) -> Dict[str, object]:
    def TensorFlow_Relu_Elu_Selu_Nadam() -> keras.Sequential:
        model = keras.Sequential([layers.BatchNormalization(),
                                  layers.Dense(TF_neurons, activation='relu', input_shape=(None, num_features)),
                                  layers.BatchNormalization(),
                                  layers.Dense(TF_neurons, activation='elu'),
                                  layers.Dropout(0.2),
                                  layers.BatchNormalization(),
                                  layers.Dense(TF_neurons, activation='selu'),
                                  layers.BatchNormalization(),
                                  layers.Dense(1)])
        optimizer = keras.optimizers.Nadam(learning_rate=TF_learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def TensorFlow_Softplus_Nadam() -> keras.Sequential:
        model = keras.Sequential([layers.BatchNormalization(),
                                  layers.Dense(TF_neurons, activation='softplus', input_shape=(None, num_features)),
                                  layers.BatchNormalization(),
                                  layers.Dropout(0.2),
                                  layers.Dense(TF_neurons, activation='softplus'),
                                  layers.BatchNormalization(),
                                  layers.Dense(TF_neurons, activation='softplus'),
                                  layers.Dense(1)])
        optimizer = keras.optimizers.Nadam(learning_rate=TF_learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def TensorFlow_Synthetic() -> keras.Sequential:
        model = keras.Sequential([layers.BatchNormalization(),
                                  layers.Dense(TF_neurons, activation='relu', input_shape=(None, num_features)),
                                  layers.BatchNormalization(),
                                  layers.Dropout(0.2),
                                  layers.Dense(TF_neurons * 2, activation='softplus'),
                                  layers.BatchNormalization(),
                                  layers.Dropout(0.2),
                                  layers.Dense(TF_neurons * 2, activation='relu'),
                                  layers.Dense(1)])
        optimizer = keras.optimizers.Adam(learning_rate=TF_learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def TensorFlow_KeraTune_1() -> keras.Sequential:
        model = keras.Sequential([layers.BatchNormalization(),
                                  layers.Dense(192, activation='selu', input_shape=(None, num_features)),
                                  layers.BatchNormalization(),
                                  layers.Dense(160, activation='selu'),
                                  layers.Dropout(0.2),
                                  layers.Dense(224, activation='relu'),
                                  layers.BatchNormalization(),
                                  layers.Dense(96, activation='relu'),
                                  layers.BatchNormalization(),
                                  layers.Dense(96, activation='softplus'),
                                  layers.Dense(1)])
        optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def TensorFlow_KeraTune_2() -> keras.Sequential:
        model = keras.Sequential([layers.BatchNormalization(),
                                  layers.Dense(288, activation='relu', input_shape=(None, num_features)),
                                  layers.Dropout(0.2),
                                  layers.Dense(416, activation='relu'),
                                  layers.Dropout(0.2),
                                  layers.Dense(160, activation='selu'),
                                  layers.Dense(256, activation='softplus'),
                                  layers.BatchNormalization(),
                                  layers.Dense(256, activation='softplus'),
                                  layers.Dense(320, activation='softplus'),
                                  layers.Dropout(0.2),
                                  layers.Dense(320, activation='relu'),
                                  layers.BatchNormalization(),
                                  layers.Dense(320, activation='softplus'),
                                  layers.Dense(1)])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def TensorFlow_KeraTune_Conv_1() -> keras.Sequential:
        model = keras.Sequential()
        input_shape = (num_features,)
        model.add(layers.BatchNormalization(),)
        model.add(keras.layers.Dense(units=34, activation='relu', input_shape=input_shape))
        model.add(keras.layers.Reshape((34, 1)))

        query = keras.layers.Input(shape=(None, 34))
        value = keras.layers.Input(shape=(None, 34))
        attention = keras.layers.Attention()
        query_transformed = keras.layers.Dense(48, activation='sigmoid')(query)
        value_transformed = keras.layers.Dense(48, activation='sigmoid')(value)
        attended_values = attention([query_transformed, value_transformed])
        attended_values = keras.layers.Input(tensor=attended_values, name='attended_values')
        model.add(attended_values)
        model.add(layers.BatchNormalization(),)
        model.add(keras.layers.Conv1D(filters=18, kernel_size=3, activation='tanh'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling1D(pool_size=2))

        model.add(keras.layers.Conv1D(filters=36, kernel_size=5, activation='tanh'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling1D(pool_size=3))
        model.add(layers.BatchNormalization(),)
        model.add(keras.layers.Conv1D(filters=54, kernel_size=3, activation='tanh'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling1D(pool_size=2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=96, activation='relu'))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Dense(1))

        optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])

        return model

    def TensorFlow_KeraTune_Conv_2() -> keras.Sequential:

        input = keras.layers.Input(shape=(34))
        x = keras.layers.BatchNormalization()(input)
        x = tf.expand_dims(input, axis=2)
        x = tf.expand_dims(x, axis=1)
        x = keras.layers.Conv1D(filters=24, kernel_size=1, activation='tanh')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1D(filters=24, kernel_size=1, activation='tanh')(x)
        x = keras.layers.Dropout(0.34849147081255716)(x)
        x = keras.layers.Conv1D(filters=32, kernel_size=1, activation='tanh')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=64, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=64, activation='relu')(x)
        x = keras.layers.Dense(units=96, activation='tanh')(x)
        output = keras.layers.Dense(1, activation='relu')(x)

        optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
        model = keras.models.Model(inputs=input, outputs=output)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])

        return model

    def TenserFlow_KeraTune_Conv_3() -> keras.Sequential:
        inputs = keras.layers.Input(shape=(None, 34))
        x1 = keras.layers.Dense(32, activation='relu')(inputs)
        reshaped_inputs = keras.layers.Reshape((2, 2, 8))(x1)
        conv_layer = keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu')(reshaped_inputs)
        conv_layer = keras.layers.Flatten()(conv_layer)
        conv_layer = keras.layers.BatchNormalization()(conv_layer)
        dense_layer = keras.layers.Dense(units=288, activation='relu')(conv_layer)
        dense_layer = keras.layers.Dense(units=288, activation='relu')(dense_layer)
        dense_layer = keras.layers.Flatten()(dense_layer)
        dense_layer = keras.layers.Dense(units=288, activation='relu')(dense_layer)
        dense_layer = keras.layers.Dense(units=288, activation='relu')(dense_layer)
        dense_layer = keras.layers.Dense(units=288, activation='relu')(dense_layer)
        outputs = keras.layers.Dense(units=1)(dense_layer)
        optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])
        return model

    def TensorFlow_KeraTune_Conv_4():
        model = keras.Sequential()
        model.add(keras.layers.Conv1D(64, kernel_size=3, activation='tanh', input_shape=(34, 1)))
        model.add(keras.layers.Conv1D(64, kernel_size=3, activation='tanh'))
        model.add(keras.layers.Conv1D(64, kernel_size=3, activation='tanh'))
        model.add(keras.layers.Conv1D(64, kernel_size=3, activation='tanh'))
        model.add(keras.layers.MaxPooling1D(pool_size=4))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(units=128, activation='sigmoid'))
        model.add(keras.layers.Dense(units=1))
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        return model

    def TensorFlow_KeraTune_Conv_1Matrix():
        model = keras.Sequential()
        model.add(MyPreprocessingLayer())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.SeparableConv1D(6, kernel_size=1, activation='tanh'))
        model.add(keras.layers.Conv1D(filters=20, kernel_size=5, activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=128, activation='elu'))
        model.add(keras.layers.Dense(units=128, activation='elu'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units=128, activation='elu'))
        model.add(keras.layers.Dense(1))

        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])

        return model

    def TensorFlow_KeraTune_Conv_2Matrix():
        model = keras.Sequential()
        model.add(layers.BatchNormalization())
        model.add(MyPreprocessingLayer())
        model.add(keras.layers.SeparableConv1D(12, kernel_size=1, activation='tanh'))
        model.add(keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=96, activation='softplus'))
        model.add(keras.layers.Dense(units=96, activation='softplus'))
        model.add(keras.layers.Dropout(0.0))
        model.add(keras.layers.Dense(units=96, activation='softplus'))
        model.add(keras.layers.Dense(1))

        optimizer = keras.optimizers.Nadam(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])

        return model

    def TensorFlow_KeraTune_Conv_3Matrix():
        model = keras.Sequential()
        model.add(keras.layers.BatchNormalization())
        model.add(MyPreprocessingLayer())
        model.add(keras.layers.Conv1D(filters=24, kernel_size=5, activation='selu'))
        model.add(keras.layers.SeparableConv1D(16, kernel_size=1, activation='selu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=96, activation='elu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(units=96, activation='elu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units=96, activation='elu'))
        model.add(keras.layers.Dense(1))

        optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

        return model

    def TensorFlow_KeraTune_Conv_4Matrix():
        model = keras.Sequential()
        model.add(keras.layers.BatchNormalization())
        model.add(MyPreprocessingLayer())
        model.add(keras.layers.SeparableConv1D(10, kernel_size=1, activation='tanh'))
        model.add(keras.layers.Conv1D(filters=20, kernel_size=1, activation='relu'))
        model.add(keras.layers.Conv1D(filters=20, kernel_size=1, activation='tanh'))
        model.add(keras.layers.Conv1DTranspose(filters=24, kernel_size=1, activation='tanh'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(units=32, activation='tanh'))
        model.add(keras.layers.Dense(units=32, activation='selu'))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(units=128, activation='softplus'))
        model.add(keras.layers.Dense(1))

        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

        return model

    def TensorFlow_KeraTune_Conv_1Flat():
        input = keras.layers.Input(shape=(34))
        x = keras.layers.BatchNormalization()(input)
        x = tf.expand_dims(x, axis=2)
        x = tf.expand_dims(x, axis=1)
        x = keras.layers.Conv1D(filters=16, kernel_size=1, activation='elu')(x)
        x = keras.layers.Conv1D(filters=24, kernel_size=1, activation='tanh')(x)
        x = keras.layers.Conv1D(filters=16, kernel_size=1, activation='selu')(x)
        x = keras.layers.Dropout(0.44417634318940014)(x)
        x = tf.squeeze(x, axis=1)
        x = keras.layers.Conv1DTranspose(filters=24, kernel_size=1, activation='elu')(x)
        x = keras.layers.Conv1DTranspose(filters=32, kernel_size=1, activation='relu')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=128, activation='softplus')(x)
        x = keras.layers.Dense(units=128, activation='elu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=128, activation='selu')(x)
        output = keras.layers.Dense(1, activation='selu')(x)

        model = keras.models.Model(inputs=input, outputs=output)
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])

        return model

    def TensorFlow_KeraTune_Conv_2Flat():
        input = keras.layers.Input(shape=(34))
        x = keras.layers.BatchNormalization()(input)
        x = tf.expand_dims(x, axis=2)
        x = tf.expand_dims(x, axis=1)
        x = keras.layers.Conv1D(filters=24, kernel_size=1, activation='tanh')(x)
        x = keras.layers.Conv1D(filters=24, kernel_size=1, activation='tanh')(x)
        x = keras.layers.Dropout(0.34849147081255716)(x)
        x = keras.layers.Conv1D(filters=32, kernel_size=1, activation='tanh')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=64, activation='relu')(x)
        x = keras.layers.Dense(units=64, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=96, activation='tanh')(x)
        output = keras.layers.Dense(1, activation='relu')(x)
        optimizer = keras.optimizers.RMSprop(learning_rate=0.001)

        model = keras.models.Model(inputs=input, outputs=output)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])
        return model

    def TensorFlow_KeraTune_Conv_3Flat():

        num_conv_layers = 4
        num_dropouts = 1
        dropout_rate = 0.1671256121594838
        num_dense_layers = 2
        l1_regularization = 0.1671256121594838
        l2_regularization = 0.29732598490943557

        input = keras.layers.Input(shape=(34))
        x = keras.layers.BatchNormalization()(input)
        x = tf.expand_dims(x, axis=2)
        x = tf.expand_dims(x, axis=1)

        # Add convolutional layers
        for _ in range(num_conv_layers):
            x = keras.layers.Conv1D(filters=32, kernel_size=1)(x)
            x = keras.layers.Activation('selu')(x)
            x = keras.layers.Dropout(0.30713858568974556)(x)
            x = keras.layers.ActivityRegularization(l1=l1_regularization, l2=l2_regularization)(x)

        # Add dropout layers
        for _ in range(num_dropouts):
            x = keras.layers.Dropout(dropout_rate)(x)

        x = tf.squeeze(x, axis=1)

        # Add transpose convolutional layers
        for _ in range(num_conv_layers):
            x = keras.layers.Conv1DTranspose(96, kernel_size=1)(x)
            x = keras.layers.Activation('relu')(x)

        x = keras.layers.Flatten()(x)

        for _ in range(num_dense_layers):
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dense(128)(x)
            x = keras.layers.Activation('selu')(x)
            x = keras.layers.Dropout(0.30582557931042254)(x)
            x = keras.layers.ActivityRegularization(l1=l1_regularization, l2=l2_regularization)(x)

        # Add remaining layers
        x = keras.layers.Flatten()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(128)(x)
        x = keras.layers.Activation('selu')(x)
        output = keras.layers.Dense(1, activation='selu')(x)

        model = keras.models.Model(inputs=input, outputs=output)

        optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])
        return model

    def TensorFlow_KeraTune_Conv_4Flat():
        input = keras.layers.Input(shape=(34))
        x = tf.expand_dims(input, axis=2)
        x = keras.layers.SeparableConv1D(filters=32, kernel_size=1, activation='softplus')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.SeparableConv1D(filters=8, kernel_size=5, activation='tanh')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1D(filters=24, kernel_size=3, activation='selu')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1D(filters=16, kernel_size=5, activation='softplus')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1DTranspose(filters=24, kernel_size=1, activation='selu')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1DTranspose(filters=24, kernel_size=1, activation='relu')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=128, activation='elu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=128, activation='selu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=64, activation='selu')(x)
        output = keras.layers.Dense(1, activation='selu')(x)
        model = keras.models.Model(inputs=input, outputs=output)
        optimizer = keras.optimizers.Nadam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])
        return model

    def TensorFlow_KeraTune_Conv_5Flat():
        input = keras.layers.Input(shape=(34))
        x = keras.layers.BatchNormalization()(input)
        x = tf.expand_dims(x, axis=2)
        x = keras.layers.SeparableConv1D(filters=8, kernel_size=5, activation='tanh')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SeparableConv1D(filters=32, kernel_size=3, activation='selu')(x)
        x = keras.layers.SpatialDropout1D(0.2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1D(filters=24, kernel_size=5, activation='tanh')(x)
        x = keras.layers.SpatialDropout1D(0.4)(x)
        x = keras.layers.SimpleRNN(units=64, return_sequences=True)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LSTM(units=64, return_sequences=True)(x)
        x = keras.layers.SpatialDropout1D(0.2)(x)
        x = keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu')(x)
        x = keras.layers.SpatialDropout1D(0.4)(x)
        x = keras.layers.SpatialDropout1D(0.4)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1DTranspose(filters=24, kernel_size=5, activation='tanh')(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=256, activation='tanh')(x)
        x = keras.layers.Dense(units=128, activation='elu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=128, activation='relu')(x)
        x = keras.layers.Dense(1, activation='elu')(x)
        output = keras.layers.Dense(1, activation='selu')(x)
        model = keras.models.Model(inputs=input, outputs=output)
        optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])
        return model

    def TensorFlow_KeraTune_Conv_6Flat():
        input = keras.layers.Input(shape=(34))

        x = keras.layers.BatchNormalization()(input)
        x = tf.expand_dims(x, axis=2)
        x = tf.expand_dims(x, axis=2)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ConvLSTM1D(filters=64, kernel_size=1)(x)
        x = keras.layers.SpatialDropout1D(rate=0.39476)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SeparableConv1D(filters=64, kernel_size=1, activation='tanh')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu')(x)
        x = keras.layers.SpatialDropout1D(rate=0.20179)(x)
        x = keras.layers.SimpleRNN(units=32, return_sequences=True)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LSTM(units=16, return_sequences=True)(x)
        x = keras.layers.SpatialDropout1D(0.068672)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1DTranspose(filters=24, kernel_size=5, activation='softplus')(x)
        x = keras.layers.Dropout(0.43681)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=64, activation='sigmoid')(x)
        x = keras.layers.Dense(units=512,activation='sigmoid')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=64, activation='softplus')(x)

        output = keras.layers.Dense(1, activation='selu')(x)

        model = keras.models.Model(inputs=input, outputs=output)

        optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])
        return model


    keras_models = {
        'TensorFlow_Relu_Elu_Selu_Nadam':
        CustomKerasRegressor(name='TensorFlow_Relu_Elu_Selu_Nadam', build_fn=TensorFlow_Relu_Elu_Selu_Nadam,
                             batch_size=TF_batch_size, epochs=TF_number_of_epochs),
        'TensorFlow_Softplus_Nadam':
        CustomKerasRegressor(name='TensorFlow_Softplus_Nadam', build_fn=TensorFlow_Softplus_Nadam,
                             batch_size=TF_batch_size, epochs=TF_number_of_epochs),
        'TensorFlow_Synthetic':
        CustomKerasRegressor(name='TensorFlow_Synthetic', build_fn=TensorFlow_Synthetic,
                             batch_size=TF_batch_size, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_1':
        CustomKerasRegressor(name='TensorFlow_KeraTune_1', build_fn=TensorFlow_KeraTune_1,
                             batch_size=TF_batch_size, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_2':
        CustomKerasRegressor(name='TensorFlow_KeraTune_2', build_fn=TensorFlow_KeraTune_2,
                             batch_size=TF_batch_size, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_Conv_1':
        CustomKerasRegressor(name='TensorFlow_KeraTune_Conv_1', build_fn=TensorFlow_KeraTune_Conv_1,
                             batch_size=TF_batch_size, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_Conv_2':
        CustomKerasRegressor(name='TensorFlow_KeraTune_Conv_2', build_fn=TensorFlow_KeraTune_Conv_2,
                             batch_size=TF_batch_size, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_Conv_3':
        CustomKerasRegressor(name='TensorFlow_KeraTune_Conv_3', build_fn=TenserFlow_KeraTune_Conv_3,
                             batch_size=TF_batch_size, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_Conv_4':
        CustomKerasRegressor(name='TensorFlow_KeraTune_Conv_4', build_fn=TensorFlow_KeraTune_Conv_4,
                             batch_size=256, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_Conv_1Matrix':
        CustomKerasRegressor(name='TensorFlow_KeraTune_Conv_1Matrix', build_fn=TensorFlow_KeraTune_Conv_1Matrix,
                             batch_size=256, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_Conv_2Matrix':
        CustomKerasRegressor(name='TensorFlow_KeraTune_Conv_2Matrix', build_fn=TensorFlow_KeraTune_Conv_2Matrix,
                             batch_size=128, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_Conv_3Matrix':
        CustomKerasRegressor(name='TensorFlow_KeraTune_Conv_3Matrix', build_fn=TensorFlow_KeraTune_Conv_3Matrix,
                             batch_size=128, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_Conv_4Matrix':
        CustomKerasRegressor(name='TensorFlow_KeraTune_Conv_4Matrix', build_fn=TensorFlow_KeraTune_Conv_4Matrix,
                             batch_size=64, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_Conv_1Flat':
        CustomKerasRegressor(name='TensorFlow_KeraTune_Conv_1Flat', build_fn=TensorFlow_KeraTune_Conv_1Flat,
                             batch_size=256, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_Conv_2Flat':
        CustomKerasRegressor(name='TensorFlow_KeraTune_Conv_2Flat', build_fn=TensorFlow_KeraTune_Conv_2Flat,
                             batch_size=256, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_Conv_3Flat':
        CustomKerasRegressor(name='TensorFlow_KeraTune_Conv_3Flat', build_fn=TensorFlow_KeraTune_Conv_3Flat,
                             batch_size=256, epochs=TF_number_of_epochs),
        'TensorFlow_KeraTune_Conv_4Flat':
        CustomKerasRegressor(name='TensorFlow_KeraTune_Conv_4Flat', build_fn=TensorFlow_KeraTune_Conv_4Flat,
                             batch_size=64, epochs=TF_number_of_epochs), #16
        'TensorFlow_KeraTune_Conv_5Flat':
        CustomKerasRegressor(name='TensorFlow_KeraTune_Conv_5Flat', build_fn=TensorFlow_KeraTune_Conv_5Flat,
                             batch_size=64, epochs=TF_number_of_epochs), #16
        'TensorFlow_KeraTune_Conv_6Flat':
        CustomKerasRegressor(name='TensorFlow_KeraTune_Conv_6Flat', build_fn=TensorFlow_KeraTune_Conv_6Flat,
                             batch_size=64, epochs=TF_number_of_epochs)} #32

    for model in keras_models:
        models_dict[model] = keras_models[model]

    return models_dict


models = {'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42, max_depth=1000),
          'DecisionTree': DecisionTreeRegressor(max_depth=1000, random_state=42),
          'KNeighbors': KNeighborsRegressor(n_neighbors=5),
          'ExtraTrees': ExtraTreesRegressor(random_state=42, n_estimators=300, max_depth=300),
          'GradientBoosting': GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, random_state=42),
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

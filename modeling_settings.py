from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.linear_model import Ridge, Lasso, ElasticNet, Lars, LassoLars, OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge, ARDRegression, LogisticRegression, SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor, HuberRegressor, RANSACRegressor
from keras.wrappers.scikit_learn import KerasRegressor, BaseWrapper
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import logging
#from typing import dict


DefaultColumns = ['DLeft', 'ops_station_lat', 'ops_station_lon', 'in_train', 'start_lat', 'start_lon']
TF_DefaultColumns = ['DLeft', 'ops_station_lat', 'ops_station_lon', ]
TF_number_of_epochs = 100
TF_batch_size = 32
TF_neurons = 256
TF_learning_rate = 0.00001
TF_input_shape = (None, )


def declare_keras_models(models_dict: dict, num_features: int) -> dict:
    class MyKerasRegressor(KerasRegressor, BaseEstimator, RegressorMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.model_ = None

        def fit(self, X, y, **kwargs):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.float32)
            history = super().fit(X, y, **kwargs)
            self.model_ = history.model  # store the trained model in the instance variable
            return self.model_
        
        def predict(self, X):
            if not isinstance(X, (tf.Tensor, tf.compat.v1.Tensor, tf.python.framework.ops.EagerTensor)):
                X = tf.convert_to_tensor(X, dtype=tf.float32) 
            return self.model_.predict(X)
     
    class CustomKerasRegressor(MyKerasRegressor, BaseWrapper):
        """Wrapper class to make KerasRegressor work with Scikit-learn's cross-validation."""
        def __init__(self, build_fn=None, **sk_params):
            self.build_fn = build_fn
            self.sk_params = sk_params

        def __call__(self, **kwargs):
            return self.build_fn(**{**self.sk_params, **kwargs})

        def get_params(self, **params):
            return {**self.sk_params, **params}
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
                                  layers.Dense(TF_neurons * 4, activation='softmax'),
                                  layers.Dense(TF_neurons * 2, activation='elu'),
                                  layers.Dense(TF_neurons, activation='relu'),
                                  layers.Dense(1)])
        optimizer = keras.optimizers.Adam(learning_rate=TF_learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    keras_models = {'TensorFlow_Relu_Elu_Selu_Nadam':
                    CustomKerasRegressor(build_fn=TensorFlow_Relu_Elu_Selu_Nadam, batch_size=TF_batch_size,
                                   epochs=TF_number_of_epochs),
                    'TensorFlow_Softplus_Nadam':
                    CustomKerasRegressor(build_fn=TensorFlow_Softplus_Nadam, batch_size=TF_batch_size,
                                   epochs=TF_number_of_epochs),
                    'TensorFlow_Synthetic':
                    CustomKerasRegressor(build_fn=TensorFlow_Synthetic, batch_size=TF_batch_size,
                                   epochs=TF_number_of_epochs)
         }
    for model in keras_models:
        models_dict[model] = keras_models[model]
    return models_dict


models = {'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42, max_depth=100),
          'DecisionTree': DecisionTreeRegressor(max_depth=1000, random_state=42),
          'KNeighbors': KNeighborsRegressor(n_neighbors=5),
          'ExtraTrees': ExtraTreesRegressor(random_state=42, n_estimators=300, max_depth=100),
          'GradientBoosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.75, random_state=42),
          'MLP': MLPRegressor(hidden_layer_sizes=(256, 256, 128), activation='relu', solver='adam', random_state=42,
                              max_iter=1000),
          'Ridge': Ridge(alpha=1.0),
          'Lasso': Lasso(alpha=1.0),
          'Lars': Lars(n_nonzero_coefs=10),
          'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
          'BayesianRidge': BayesianRidge(),
          'ARDRegression': ARDRegression(),
          # 'LogisticRegression': LogisticRegression(),
          'SGDRegressor': SGDRegressor(),
          'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
          'HuberRegressor': HuberRegressor(),
          'RANSACRegressor': RANSACRegressor(),
          'ElasticNet': ElasticNet(),
          'LassoLars': LassoLars(),
          'AdaBoost': AdaBoostRegressor(random_state=42, n_estimators=300)}

# TF_models_list = [model for model in models if model.startswith('TensorFlow')]

sklearn_list = ['RandomForest', 'DecisionTree', 'KNeighbors', 'ExtraTrees', 'GradientBoosting', 'MLP',
                'GaussianProcess_RBF', 'GaussianProcess_Matern', 'GaussianProcess_RQ', 'GaussianProcess_Exp',
                'SVR_linear', 'SVR_poly', 'SVR_rbf', 'SVR_sigmoid', 'AdaBoost', 'Ridge', 'Lasso', 'ElasticNet',
                'Lars', 'LassoLars', 'OrthogonalMatchingPursuit', 'BayesianRidge', 'ARDRegression',
                'LogisticRegression', 'SGDRegressor', 'PassiveAggressiveRegressor', 'HuberRegressor',
                'RANSACRegressor']

DefaultTrainingDateCut = '2023-05-15'
filter_types = ['savgol', 'butter', 'none']
filter_type = filter_types[1]

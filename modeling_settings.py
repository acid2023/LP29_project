from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, Lars, LassoLars, OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge, ARDRegression, PassiveAggressiveRegressor, RANSACRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin

from tensorflow import keras
from keras import layers
from keras.wrappers.scikit_learn import KerasRegressor

import pandas as pd

from typing import Dict

DefaultTrainingDateCut = '2023-05-15'
filter_types = ['savgol', 'butter', 'none']
filter_type = filter_types[2]

DefaultColumns = ['DLeft', 'ops_station_lat', 'ops_station_lon', 'update']

TF_DefaultColumns = ['DLeft', 'ops_station_lat', 'ops_station_lon', 'update']
TF_number_of_epochs = 150
TF_batch_size = 32
TF_neurons = 512
TF_learning_rate = 0.0001
TF_input_shape = (None, )


class CustomKerasRegressor(KerasRegressor, BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> keras.Sequential:
        history = super().fit(X, y, **kwargs)
        self.model_ = history.model
        return self.model_


def declare_keras_models(models_dict: Dict[str, object], num_features: int) -> Dict[str, object]:
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
                                  layers.Dense(TF_neurons, activation='relu'),
                                  layers.Dense(1)])
        optimizer = keras.optimizers.Nadam(learning_rate=TF_learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    keras_models = {
        'TensorFlow_Relu_Elu_Selu_Nadam':
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
          'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
          'RANSACRegressor': RANSACRegressor(),
          'ElasticNet': ElasticNet(),
          'LassoLars': LassoLars(),
          'AdaBoost': AdaBoostRegressor(random_state=42, n_estimators=300)}


sklearn_list = list(models.keys())

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, Exponentiation
# from sklearn.svm import SVR
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.losses as losses


DefaultColumns = ['DLeft', 'ops station', 'o_road', 'start', 'update_month', 'update_day']
TF_DefaultColumns = ['DLeft', 'in_train', 'update_month', 'start']
TF_number_of_epochs = 21
TF_batch_size = 64
TF_neurons = 256
gaussian_kernel = [RBF(length_scale=1.0), Matern(length_scale=1.0, nu=1.5),
                   RationalQuadratic(length_scale=1.0, alpha=1.5), Exponentiation(RBF(length_scale=1.0), exponent=2)]
SVR_kernel = ['linear', 'poly', 'rbf', 'sigmoid']
models = {'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42, max_depth=100),
          'DecisionTree': DecisionTreeRegressor(max_depth=1000, random_state=42),
          'KNeighbors': KNeighborsRegressor(n_neighbors=5),
          'ExtraTrees': ExtraTreesRegressor(random_state=42, n_estimators=300, max_depth=100),
          'GradientBoosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.75, random_state=42),
          'MLP': MLPRegressor(hidden_layer_sizes=(256, 256, 128), activation='relu', solver='adam', random_state=42,
                              max_iter=1000),
          # 'GaussianProcess_RBF': GaussianProcessRegressor(gaussian_kernel[0]),
          # 'GaussianProcess_Matern': GaussianProcessRegressor(gaussian_kernel[1]),
          # 'GaussianProcess_RQ': GaussianProcessRegressor(gaussian_kernel[2]),
          # 'GaussianProcess_Exp': GaussianProcessRegressor(gaussian_kernel[3]),
          # 'SVR_linear': SVR(kernel=SVR_kernel[0], gamma=0.1),
          # 'SVR_poly': SVR(kernel=SVR_kernel[1], gamma=0.1),
          # 'SVR_rbf': SVR(kernel=SVR_kernel[2], gamma=0.1),
          # 'SVR_sigmoid': SVR(kernel=SVR_kernel[3], gamma=0.1, coef0=0.0),
          'AdaBoost': AdaBoostRegressor(random_state=42, n_estimators=300),
          'TensorFlow_Relu_Elu_Selu_Nadam': keras.Sequential([layers.Dense(TF_neurons, activation='relu',
                                                              input_shape=[len(TF_DefaultColumns)]),
                                                              layers.Dense(TF_neurons, activation='elu'),
                                                              layers.Dense(TF_neurons, activation='selu'),
                                                              layers.Dense(1)]),
          'TensorFlow_Softplus_Nadam': keras.Sequential([layers.Dense(TF_neurons, activation='softplus',
                                                                      input_shape=[len(TF_DefaultColumns)]),
                                                         layers.Dense(TF_neurons, activation='softplus'),
                                                         layers.Dense(TF_neurons, activation='softplus'),
                                                         layers.Dense(TF_neurons, activation='softplus'),
                                                         layers.Dense(TF_neurons, activation='softplus'),
                                                         layers.Dense(1)]),
          'TensorFlow_Synthetic': keras.Sequential([layers.Dense(TF_neurons, activation='relu',
                                                                 input_shape=[len(TF_DefaultColumns)]),
                                                    layers.Dense(TF_neurons * 2, activation='softplus'),
                                                    layers.Dense(TF_neurons * 2, activation='selu'),
                                                    layers.Dense(TF_neurons * 2, activation='softmax'),
                                                    layers.Dense(TF_neurons, activation='elu'),
                                                    layers.Dense(TF_neurons, activation='relu'),
                                                    layers.Dense(1)])}
TF_models_list = ['TensorFlow_Relu_Elu_Selu_Nadam', 'TensorFlow_Softplus_Nadam', 'TensorFlow_Synthetic']
TF_optimizers = {'TensorFlow_Relu_Elu_Selu_Nadam': keras.optimizers.Nadam(learning_rate=0.001),
                 'TensorFlow_Softplus_Nadam': keras.optimizers.Nadam(learning_rate=0.001),
                 'TensorFlow_Synthetic': keras.optimizers.Adam(learning_rate=0.001)}
TF_metrics = [tf.keras.metrics.MeanAbsoluteError(name="mae")]
TF_loss = {'TensorFlow_Relu_Elu_Selu_Nadam': losses.Huber(),
           'TensorFlow_Softplus_Nadam': losses.MeanAbsoluteError(),
           'TensorFlow_Synthetic': losses.MeanAbsoluteError()}
sklearn_list = ['RandomForest', 'DecisionTree', 'KNeighbors', 'ExtraTrees', 'GradientBoosting', 'MLP',
                'GaussianProcess_RBF', 'GaussianProcess_Matern', 'GaussianProcess_RQ', 'GaussianProcess_Exp',
                'SVR_linear', 'SVR_poly', 'SVR_rbf', 'SVR_sigmoid', 'AdaBoost']
DefaultTrainingDateCut = '2023-05-01'
filter_types = ['savgol', 'butter', 'none']
filter_type = filter_types[0]

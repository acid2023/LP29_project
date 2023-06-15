import pandas as pd
from scipy.signal import savgol_filter, filtfilt, butter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
import pickle
from typing import Dict, List
import logging
import modeling_settings as mds
import folders
from folders import folder_check


def save_models(models: Dict):
    path = folder_check(folders.models_folder)
    model_filename = f'{path}models_sklearn.pkl'
    sklearn_models = {}
    models_list = []
    for model_name, model in models[0].items():
        if model_name in mds.TF_models_list:
            models_list.append(model_name)
            ts_filename = f'{path}{model_name}.h5'
            model.save(ts_filename)
        else:
            sklearn_models[model_name] = model
    with open(model_filename, 'wb') as file:
        pickle.dump(sklearn_models, file)
    with open(f'{path}models_list.pkl', 'wb') as file:
        pickle.dump(models_list, file)
    with open(f'{path}models_metrics.pkl', 'wb') as file:
        pickle.dump(models[1], file)
    with open(f'{path}models_scores.pkl', 'wb') as file:
        pickle.dump(models[2], file)
    with open(f'{path}columns_list.pkl', 'wb') as file:
        pickle.dump(models[3], file)


def load_models() -> Dict:
    path = folder_check(folders.models_folder)
    model_filename = f'{path}models_sklearn.pkl'
    with open(model_filename, 'rb') as file:
        models = pickle.load(file)
    for model_name in mds.TF_models_list:
        ts_filename = f'{path}{model_name}.h5'
        model = keras.models.load_model(ts_filename)
        models[model_name] = model
    return models


def preprocessing_trains(trains: pd.DataFrame) -> pd.DataFrame:
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


def preprocessing_update_trains(update_trains: pd.DataFrame) -> pd.DataFrame:
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


def get_models_metrics(models: Dict, metrics_data: pd.DataFrame) -> pd.DataFrame:
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


def create_models(df: pd.DataFrame, columns_list: List) -> List:
    logging.info('Started preprocessing')
    trains = preprocessing_trains(df)
    logging.info('Preprocessing done')
    trains['target'] = smooth_data(trains['to_home'], mds.filter_type)
    logging.info('Filtering done')
    logging.info('Fitting models')
    logging.info(columns_list)
    fit_models = {}
    metrics_data = {}
    for name, model in mds.models.items():
        logging.info(f'fitting model {name} started')
        if name in mds.TF_models_list:
            X = trains[mds.TF_DefaultColumns]
        elif name in mds.sklearn_list:
            X = trains[columns_list]
        y = trains['target']
        X_train, X_remains, y_train, y_remains = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_remains, y_remains, test_size=0.5, random_state=42)
        metrics_data[name] = [X_test, y_test]
        if name in mds.TF_models_list:
            model.compile(loss=mds.TF_loss[name], optimizer=mds.TF_optimizers[name], metrics=mds.TF_metrics)
            mds.TF_optimizers[name].build(model.trainable_variables)
            model.fit(X_train, y_train, epochs=mds.TF_number_of_epochs,
                      batch_size=mds.TF_batch_size, validation_data=(X_val, y_val))
            fit_models[name] = model
        elif name in mds.sklearn_list:
            fit_models[name] = model.fit(X_train, y_train)
        logging.info(f'fitting model {name} finished')
    logging.info('All models fitted')
    logging.info('Calculating metrics')
    metrics = get_models_metrics(fit_models, metrics_data)
    logging.info('Metrics calculated')
    logging.info(f"Metrics: \n{metrics.to_string(index=True, line_width=80)}")
    logging.info('Calculating scores')
    scores = cross_validation_test(fit_models, metrics_data)
    logging.info('Scores calculated')
    logging.info(f"Scores: \n{scores.to_string(index=True, line_width=80)}")
    return [fit_models, metrics, scores, columns_list]


def prediction(df: pd.DataFrame, models: Dict, columns_list: List) -> pd.DataFrame:
    logging.info('Started preprocessing')
    update_trains = preprocessing_update_trains(df)
    logging.info('Preprocessing done')
    logging.info('Predicting')
    columns_to_keep = []
    for name, model in models.items():
        logging.info(f'predicting for model {name} started')
        if name in mds.sklearn_list:
            update_X = update_trains[columns_list]
        elif name in mds.TF_models_list:
            update_X = update_trains[mds.TF_DefaultColumns]
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
    update_trains = update_trains[columns_to_keep]
    return update_trains


if __name__ == "__main__":
    pass

import pandas as pd
import numpy as np
import datetime

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

import pickle
import os

import folders
from folders import folder_check
import osm


def preprocessing_distance(df):
    if 'осталоcь расстояние' in df.columns:
        df = df.rename(columns = {'осталоcь расстояние': 'distance'})        
    df_features = df[df['статус test'] != '4. у покупателя'].copy()
    df_features.loc[df_features.dest == df_features['ops station'], 'distance'] = 0
    df_features.in_train = df_features.in_train.fillna(1)
    for start_id in df_features[df_features.distance.isna()].start_id.unique():
        dfnl = df_features[df_features.start_id == start_id].sort_values('update')
        dfnl['distance'] = dfnl['distance'].interpolate()
        df_features.loc[dfnl.index, 'distance'] = dfnl['distance']
    distance_mask = df_features['distance'].isna() & ~df_features['DLeft'].isna()
    dest_mask = df_features.dest == 'ЛЕНА-ВОСТОЧНАЯ (927800)'
    df_features.loc[distance_mask & dest_mask, 'distance'] = df_features.loc[distance_mask & dest_mask, 'DLeft']
    # df_features.drop(columns=['start_id', 'DLeft'], inplace=True)
    # df_features.reset_index(drop=True, inplace=True)
    df_features = df_features.sort_index()
    #df_features = df_features.rename(columns={'distance': 'DLeft'})
    df_features['DLeft'] = df_features['DLeft'].fillna(df_features['distance'])
    return df_features.drop(columns=['distance', 'start_id'])


def preprocessing_roads(df):
    df_sorted = df.sort_values(by=['_num', 'update'])
    df_sorted['o_road'] = df_sorted.groupby('_num')['o_road'].ffill().bfill()
    df_sorted = df_sorted.sort_index()
    return df_sorted

def initial_preprocessing(df):
    df = df.rename(columns = {'осталоcь расстояние': 'distance'})
    df = preprocessing_roads(df)
    df['in_train' ]= df['in_train'].fillna(1)
    ops_mask = df['ops'] == 'бросание поезда на станции'
    df.loc[ops_mask, 'ops'] = 1
    df.loc[~ops_mask, 'ops'] = 0
    df['ops'] = df['ops'].astype('float')
    df_work = df[['DLeft', 'start_id', 'distance', 'ops station', 'start', 'dest', 'update', 'in_train', 'o_road', '_delivery', 'статус test', 'ops']].copy()
    return preprocessing_distance(df_work)


def get_no_leak_stay(df, update_cut, **kwargs):
    predict = kwargs.get('predict', False)
    client = df[df['статус test'] == '4. у покупателя'].copy()[['dest', 'update', 'in_train', 'at_client', 'start_id']].dropna()
    if predict:
        update = client[client['update'] >= update_cut]
        return update
    mask = client.groupby('start_id')['update'].transform(lambda x: all(x < update_cut))
    train = client[mask].drop(columns=['start_id']).sample(frac=1).reset_index(drop=True)
    test =  client[~mask].drop(columns=['start_id']).sample(frac=1).reset_index(drop=True)
    return train, test

def get_no_leak(df_features, update_cut, **kwargs):
    predict = kwargs.get('predict', False)
    df_no_leak = df_features.copy()
    if predict:
        update = df_no_leak[(df_no_leak['update'] >= update_cut)].drop(columns=['_delivery', 'статус test'])#.sample(frac=1).reset_index(drop=True)
        return update
    else:
        df_no_leak = df_no_leak.dropna(subset=['_delivery'])
        df_no_leak.loc[:,'target'] =  df_no_leak['_delivery'] - df_no_leak['update']
        df_no_leak.target = df_no_leak.target.dt.days
        df_no_leak.loc[df_no_leak['ops station'] == df_no_leak['dest'], 'target'] = 0
        df_no_leak.drop(df_no_leak[(df_no_leak['target'] < 0)].index, inplace=True)
        df_no_leak.reset_index(drop=True)
        train = df_no_leak[(df_no_leak['update'] <=update_cut) & (df_no_leak['_delivery']<=update_cut)].drop(columns=['_delivery', 'статус test']).sample(frac=1).reset_index(drop=True)
        test = df_no_leak[(df_no_leak['update'] > update_cut)].drop(columns=['_delivery', 'статус test']).sample(frac=1).reset_index(drop=True)
        return train, test


def to_datetime_days(days_timestamp):
    return datetime.datetime.fromtimestamp(days_timestamp * 86400)


def to_timestamp_days(date):
    return int(datetime.datetime.timestamp(date) / 86400)


def preprocessing(df, **kwargs) -> pd.DataFrame:
    reset = kwargs.get('reset', False)
    if 'DLeft' in df.columns:
        df.dropna(subset=['DLeft'], inplace=True)
        if reset: df.reset_index(drop=True)
    if 'ops station' in df.columns:
        df.dropna(subset=['ops station'], inplace=True)
        if reset: df.reset_index(drop=True)

    print('starting coding stations')
    if 'ops station'in df.columns:
        df['ops_station_lat'], df['ops_station_lon']  = zip(*df['ops station'].apply(lambda x: osm.fetch_coordinates(x)))
        df[['ops_station_lat', 'ops_station_lon']] = df[['ops_station_lat', 'ops_station_lon']].astype(float)
        df.drop(['ops station'], axis=1, inplace=True)
        
    if 'start' in df.columns:
        df['start_lat'], df['start_lon'] = zip(*df['start'].apply(lambda x: osm.fetch_coordinates(x)))
        df[['start_lat', 'start_lon']] = df[['start_lat', 'start_lon']].astype(float)
        df.drop(['start'], axis=1, inplace=True)
        
    if 'dest' in df.columns:
        df['dest_lat'], df['dest_lon'] = zip(*df['dest'].apply(lambda x: osm.fetch_coordinates(x)))
        df[['dest_lat', 'dest_lon']] = df[['dest_lat', 'dest_lon']].astype(float)
        df.drop(['dest'], axis=1, inplace=True)    
    if reset: df.reset_index(drop=True)
    osm.save_coordinates_dict()
    print('finished coding stations')
    print('converting update times')
    if 'update' in df.columns:
        df['update'] = pd.to_datetime(df['update']).apply(to_timestamp_days)
    print('finished converting update times')
    return df


def save_one_hot_encoder(encoder: OneHotEncoder, direction):
    path = folder_check(f'{folder_check(folders.encoder_folder)}{direction}/')
    with open(f'{path}oh_encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)


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
    with open ('oh_encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)
    return encoder


def get_one_hot_encoder(df):
    filename = 'oh_encoder.pkl'
    if os.path.isfile(filename):
        with open(filename, 'rb') as file:
            encoder = pickle.load(file)
    else:
        encoder = one_hot_encoder_training(df)
        with open (filename, 'wb') as file:
            pickle.dump(encoder, file)
    return encoder



def one_hot_encoding(df: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    encoded = encoder.transform(df[['o_road']].values.reshape(-1, 1))
    encoded_df = pd.DataFrame(encoded, columns=[f"o_road_{col}" for col in encoder.get_feature_names_out()], index=df.index)
    if 'o_road_x0_nan' not in encoded_df.columns:
        encoded_df.loc[:, 'o_road_x0_nan'] = 0
    return df.join(encoded_df).drop(columns=['o_road'])



def PCA_training(df, direction):
    path = folder_check(f'{folder_check(folders.encoder_folder)}{direction}/')
    scaler = StandardScaler()
    scaler.fit(df)
    with open(f'{path}scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    pca = PCA()
    pca.fit(scaled_df)
    with open(f'{path}pca.pkl', 'wb') as file:
        pickle.dump(pca, file)


def PCA_encoding(df, direction):
    with open(f'{direction}/{direction}scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open(f'{direction}/{direction}pca.pkl', 'rb') as file:
        pca = pickle.load(file)
    scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    df_PCA = pca.transform(scaled_df)
    component_names = [f"PC{i+1}" for i in range(df_PCA.shape[1])]
    df_PCA = pd.DataFrame(df_PCA, columns=component_names, index=df.index)
    return df.join(df_PCA)


def set_input_index(df):
    if 'осталоcь расстояние' in df.columns:
        df['осталоcь расстояние'] = df['осталоcь расстояние'].apply(lambda x: float(x.replace('\xa0', '')) if isinstance(x, str) else x)
        df['осталоcь расстояние'] = df['осталоcь расстояние'].astype(float)
    df['upd_id'] =  df['_num'].astype(str) + ' ' + df['update'].astype(str)
    df.set_index(['upd_id'], inplace=True)
    return df

def get_distance(data):
    df = data.copy()
    HS = 'ЛЕНА-ВОСТОЧНАЯ (927800)'
    HS_lat, HS_lon = osm.fetch_coordinates(HS)
    if 'осталоcь расстояние' in df.columns:
        df = df.rename(columns = {'осталоcь расстояние': 'distance'})     
    df['old_distance'] = df['distance']
    df['target_lat'], _ = zip(*df.dest.map(osm.fetch_coordinates))
    df['target_DL'] = df['target_lat'].map(dest_dict)
    df_features = df.copy()
    df_features.loc[df_features.dest == df_features['ops station'], 'distance'] = 0
    
    for start_id in df_features[df_features.distance.isna()].start_id.unique():
        dfnl = df_features[df_features.start_id == start_id].sort_values('update')
        dfnl['distance'] = dfnl['distance'].interpolate()
        df_features.loc[dfnl.index, 'distance'] = dfnl['distance']
    
    distance_mask = df_features['distance'].isna() & ~df_features['DLeft'].isna()
    
    dest_mask = df_features.dest == 'ЛЕНА-ВОСТОЧНАЯ (927800)'
    
    df_features.loc[ distance_mask & dest_mask, 'distance'] = df_features.loc[distance_mask & dest_mask, 'DLeft']
    
    df_features.loc[~dest_mask, 'distance'] = df_features.loc[~dest_mask, 'distance'] + df_features.loc[~dest_mask, 'target_DL']
    df_features = df_features.sort_index()
    df_features['DLeft'] = df_features['distance']
    return df_features


def set_difference(df):
    lat, lon = osm.fetch_coordinates('ЛЕНА-ВОСТОЧНАЯ (927800)')
    df['lat_diff'] = df.apply(lambda x: abs(x['dest_lat'] - x['ops_station_lat']) if x['dest_lat'] == lat else abs(x['dest_lat'] - x['ops_station_lat']) + abs(x['dest_lat'] - lat), axis=1)
    df['lon_diff'] = df.apply(lambda x: abs(x['dest_lon'] - x['ops_station_lon']) if x['dest_lon'] == lat else abs(x['dest_lon'] - x['ops_station_lon']) + abs(x['dest_lon'] - lat), axis=1)
    return df


def getting_two_way_data(data, update_cut, **kwargs):
    training = kwargs.get('training', True)
    HS = 'ЛЕНА-ВОСТОЧНАЯ (927800)'
    HS_lat, HS_lon = osm.fetch_coordinates(HS)
    df = data[~((data.dest == HS) & (data.start == HS))].copy()
    mask = df['статус test'].str[0].isin(list('345'))
    df = df[mask]
    df = get_distance(df)
    diff = kwargs.get('diff', False)
    just_PCA = kwargs.get('just_PCA', False)
    #df.dropna(subset=['distance'], inplace=True)
    df = df[['DLeft', 'ops station', 'start', 'dest', 'arrival', 'update']]
    if training:
        df_no_leak = df.dropna(subset=['arrival']).copy()
        df_no_leak.loc[:, 'target'] =  df_no_leak.loc[:, 'arrival']
        df_no_leak['target'] =  df_no_leak['target'] - df_no_leak['update']
        df_no_leak['target'] =  df_no_leak['target'].dt.days
        df_no_leak.loc[(df_no_leak['ops station'] == df_no_leak['dest']) & (df_no_leak['dest'] == HS), 'target'] = 0
        df_no_leak.drop(df_no_leak[(df_no_leak['target'] < 0)].index, inplace=True)
       
        train = df_no_leak[(df_no_leak['update'] <=update_cut) & (df_no_leak['arrival']<=update_cut)].drop(columns=['arrival', 'update']).sample(frac=1)#.reset_index(drop=True)
        test = df_no_leak[(df_no_leak['update'] > update_cut)].drop(columns=['arrival', 'update']).sample(frac=1)#.reset_index(drop=True)
        
        coded_train = preprocessing(train.copy(), reset=False)
        coded_test = preprocessing(test.copy(), reset=False)
        
        if diff:
            coded_train = set_difference(coded_train)
            coded_test = set_difference(coded_test)

        PCA_training(coded_train.copy().drop(columns=['target']), 'full')
        
        PCA_train_encoded = PCA_encoding(coded_train.copy().drop(columns=['target']), 'full')
        PCA_test_encoded = PCA_encoding(coded_test.copy().drop(columns=['target']), 'full')

        PCA_train = pd.concat([PCA_train_encoded, coded_train[['target']]], axis=1)
        PCA_test = pd.concat([PCA_test_encoded, coded_test[['target']]], axis=1)

        train = PCA_train.drop(columns=[col for col in PCA_train.columns if col.startswith('PC')])
        test = PCA_test.drop(columns=[col for col in PCA_test.columns if col.startswith('PC')])
        
        if just_PCA:
            PCA_train.drop(columns=[col for col in PCA_train_encoded.columns if not col.startswith('PC')], inplace=True)
            PCA_test.drop(columns=[col for col in PCA_test_encoded.columns if not col.startswith('PC')], inplace=True)
        return PCA_train, PCA_test, train, test
    else:
                
        forecast = df[(df['update'] >= update_cut)].drop(columns=['arrival','update'])
         
        forecast = preprocessing(forecast, reset=False)
            
        forecast_PCA = PCA_encoding(forecast, 'full')
        
        forecast_no_PCA = forecast_PCA.drop(columns=[col for col in forecast_PCA.columns if col.startswith('PC')])

        return forecast_PCA, forecast_no_PCA

def getting_data(data, update_cut, **kwargs):
    
    training = kwargs.get('training', True)
    direction = kwargs.get('direction', 'two_way')
    HS = 'ЛЕНА-ВОСТОЧНАЯ (927800)'
    if direction == 'there':
        df = data[(data.start == HS) & (data.dest != HS)].copy()
    elif direction == 'back':
        df = data[(data.dest == HS) & (data.start != HS)].copy()
    elif direction == 'two_way':
        df = data[~((data.dest == HS) & (data.start == HS))].copy()
    df_features = initial_preprocessing(df.copy())

    if training:
        
        train, test = get_no_leak(df_features.copy(), update_cut)

        coded_train = preprocessing(train.copy())
        coded_test = preprocessing(test.copy())
        
        one_hot_encoder = one_hot_encoder_training(df.copy())
        
        PCA_training(coded_train.copy().drop(columns=['target', 'o_road']), direction)

        PCA_train_encoded = PCA_encoding(coded_train.copy().drop(columns=['target', 'o_road']), direction)
        
        PCA_train = pd.concat([PCA_train_encoded, coded_train[['target', 'o_road']]], axis=1)

        PCA_test_encoded = PCA_encoding(coded_test.copy().drop(columns=['target', 'o_road']), direction)
        
        PCA_test = pd.concat([PCA_test_encoded, coded_test[['target', 'o_road']]], axis=1)

        OH_PCA_train = one_hot_encoding(PCA_train.copy(), one_hot_encoder)
        OH_PCA_test = one_hot_encoding(PCA_test.copy(), one_hot_encoder)

        OH_train = OH_PCA_train.drop(columns=[col for col in OH_PCA_train.columns if col.startswith('PC')])
        OH_test = OH_PCA_test.drop(columns=[col for col in OH_PCA_test.columns if col.startswith('PC')])

        return OH_PCA_train, OH_PCA_test, OH_train, OH_test
    
    else:
        
        one_hot_encoder = get_one_hot_encoder(df.copy())

        update = get_no_leak(df_features.copy(), update_cut, predict=True)        
        
        update = preprocessing(update.copy(), reset=False)
             
        PCA_update_encoded = PCA_encoding(update.copy().drop(columns=['o_road']), direction)
        
        PCA_update = pd.concat([PCA_update_encoded, update[['o_road']]], axis=1)
        
        OH_PCA_update = one_hot_encoding(PCA_update.copy(), one_hot_encoder)
        
        OH_update = OH_PCA_update.drop(columns=[col for col in OH_PCA_update.columns if col.startswith('PC')])
        
        return OH_PCA_update, OH_update


def get_new_data_back(predict, model, **kwargs):
    global mapping_dict
    stay = pd.read_pickle('stay.pkl').set_index('dest_lat').drop(columns=['dest_lon']).to_dict()
    PCA_status = kwargs.get('PCA', True)
    direction = kwargs.get('direction', 'two_way')
    if PCA_status:
        dframe = predict[0].copy()
    else:
        dframe = predict[1].copy()
    no_OH = kwargs.get('no_OH', False)
    if no_OH:
        columns_to_drop = [col for col in dframe.columns if col.startswith('o_road')]
        dframe.drop(columns=columns_to_drop, inplace=True)

    forecast = dframe.copy().drop(columns=[col for col in dframe.columns if col.startswith('PC')])

    forecast['predict'] = pd.DataFrame(model.predict(dframe, batch_size=512), index=dframe.index)
    forecast['predict'] = forecast['predict'].apply(lambda x: x + 1 if x >= 0 else 1)  
    
    forecast['start_lat'], forecast['start_lon'] = forecast['dest_lat'], forecast['dest_lon']
    forecast['dest_lat'], forecast['dest_lon'] = osm.fetch_coordinates('ЛЕНА-ВОСТОЧНАЯ (927800)')
    
    if not no_OH:
        OH_columns = [col for col in forecast.columns if col.startswith('o_road_x0')]
        forecast['o_road'] = forecast[OH_columns].idxmax(axis=1).str[len('o_road_x0_'):]
        forecast.drop(columns=[col for col in forecast.columns if col.startswith('o_road_x0_')], inplace=True)
    forecast['o_road'] = forecast['start_lat'].map(mapping_dict)

    forecast['delivery'] = pd.to_timedelta(forecast['predict'], unit='D')
    forecast['_num'], forecast['update'] = zip(*forecast.index.str.split(' ')) 
    forecast['update'] = pd.to_datetime(forecast['update'])
    forecast['expected'] = forecast['update'] + forecast['delivery']
    forecast['expected'] = forecast.apply(lambda x: x['expected'] + pd.to_timedelta(stay.get(x['start_lat'], 7), unit='D'), axis=1)

    forecast['update'] = forecast['expected'].apply(to_timestamp_days)
    forecast.drop(columns=['delivery', 'expected', '_num', 'predict'], inplace=True)
    
    forecast['ops_station_lat'], forecast['ops_station_lon'] = forecast['start_lat'], forecast['start_lon']
    PCA_encoded = PCA_encoding(forecast.copy().drop(columns=['o_road']), direction)
    forecast = PCA_encoded.join(forecast[['o_road']])
    one_hot_encoder = get_one_hot_encoder(forecast)

    forecast = one_hot_encoding(forecast.copy(), one_hot_encoder)
    
    mask = forecast['start_lat'] == osm.fetch_coordinates('ПЕРЕВОЗ (923000)')[0]
    forecast['new_DL'] = forecast.groupby('start_lat')['DLeft'].transform('max')
    forecast.loc[mask, 'new_DL'] = 1139
    forecast['DLeft'] = forecast['new_DL']
    forecast.drop(columns=['new_DL'], inplace=True)
    
    OH_forecast = forecast.copy().drop(columns=[col for col in forecast.columns if col.startswith('PC')])
    
    return forecast, OH_forecast


def predict_arrival(predict, model, **kwargs):
    PCA_status = kwargs.get('PCA', True)
    direction = kwargs.get('direction', 'two_way')
    if PCA_status:
        dframe = predict[0].copy()
        columns_to_drop = [col for col in dframe.columns if col.startswith('PC') or col.startswith('o_road')]
        PCA_encoded = PCA_encoding(dframe.copy().drop(columns=columns_to_drop), direction)
        for col in PCA_encoded.columns:
            dframe[col] = PCA_encoded[col]
    else:
        dframe = predict[1].copy()
    no_OH = kwargs.get('no_OH', False)
    columns_to_drop = [col for col in dframe.columns if col.startswith('o_road')]
    dframe.drop(columns=columns_to_drop, inplace=True)
    if not no_OH:
        one_hot_encoder = one_hot_encoder_training(dframe.copy())       
        dframe['o_road'] = dframe['ops_station_lat'].map(mapping_dict)
        dframe = one_hot_encoding(dframe.copy(), one_hot_encoder)
    forecast = dframe.copy() 
    forecast['duration'] = pd.DataFrame(model.predict(dframe, batch_size=512), index=dframe.index)
    forecast['duration'] = forecast['duration'].apply(lambda x: x + 2 if x >= 0 else 2)
    _, forecast['update'] = zip(*forecast.index.str.split(' ')) 
    forecast['update'] = pd.to_datetime(forecast['update'])
    forecast['duration'] = pd.to_timedelta(forecast['duration'], unit='D')
    forecast['arrival'] = forecast['update'] + forecast['duration']
    forecast.drop(columns=dframe.columns, inplace=True)
    forecast['_num'], forecast['update'] = zip(*forecast.index.str.split(' ')) 
    forecast['update'] = pd.to_datetime(forecast['update'])
    forecast['arrival'] = forecast['arrival'].dt.date
    return forecast.drop(columns=['duration'])


def full_predict_arrival(predict, model, **kwargs):
    PCA_status = kwargs.get('PCA', True)
    if PCA_status:
        dframe = predict[0].copy()
    else:
        dframe = predict[1].copy()
    forecast = dframe.copy() 
    forecast['duration'] = pd.DataFrame(model.predict(dframe, batch_size=512), index=dframe.index)
    forecast['duration'] = forecast['duration'].apply(lambda x: x + 2 if x >= 0 else 2)
    _, forecast['update'] = zip(*forecast.index.str.split(' ')) 
    forecast['update'] = pd.to_datetime(forecast['update'])
    forecast['duration'] = pd.to_timedelta(forecast['duration'], unit='D')
    forecast['arrival'] = forecast['update'] + forecast['duration']
    forecast.drop(columns=dframe.columns, inplace=True)
    forecast['_num'], forecast['update'] = zip(*forecast.index.str.split(' ')) 
    forecast['update'] = pd.to_datetime(forecast['update'])
    forecast['arrival'] = forecast['arrival'].dt.date
    return forecast.drop(columns=['duration'])

def merge_back_data(data_1, data_2):
    back_PCA = pd.concat([data_1[0], data_2[0]], axis=0)#.reset_index(drop=True)
    back_no_PCA = back_PCA.copy().drop(columns=[col for col in back_PCA.columns if col.startswith('PC')])
    return back_PCA, back_no_PCA
   

def getting_stay_data(data, update_cut, **kwargs):
    global mapping_dict, dest_dict
    stay = pd.read_pickle('stay.pkl').set_index('dest_lat').drop(columns=['dest_lon'])
    stay = dict(zip(stay.index, stay.stay))
    direction = kwargs.get('direction', 'two_way')
    PCA_status = kwargs.get('PCA', True)
    forecast = preprocessing_roads(data)
    forecast = get_no_leak_stay(forecast, update_cut, predict=True)
    forecast = preprocessing(forecast)
    forecast['DLeft'] = forecast['dest_lat'].map(dest_dict)
    forecast['start_lat'], forecast['start_lon'] = forecast['dest_lat'], forecast['dest_lon']
    forecast['ops_station_lat'], forecast['ops_station_lon'] = forecast['dest_lat'], forecast['dest_lon']
    forecast['dest_lat'], forecast['dest_lon'] = osm.fetch_coordinates('ЛЕНА-ВОСТОЧНАЯ (927800)') 
    forecast['ops'] = 0


    forecast['o_road'] = forecast['start_lat'].map(mapping_dict)
    columns_order = ['DLeft', 'update', 'in_train', 'ops', 
                     'ops_station_lat','ops_station_lon', 
                     'start_lat', 'start_lon', 'dest_lat', 'dest_lon', 'o_road']

    forecast['_num'], forecast['update'] = zip(*forecast.index.str.split(' ')) 
    forecast['in_train'] = 1
    forecast['expected'] = pd.to_datetime(forecast['update'])

    forecast['expected'] = forecast.apply(lambda x: x['expected'] + pd.to_timedelta(stay.get(x['start_lat'], 7), unit='D'), axis=1)
    forecast['update'] = forecast['expected'].apply(to_timestamp_days)

    forecast.drop(columns=['expected', '_num'], inplace=True)

    forecast = forecast.reindex(columns=columns_order)
    PCA_encoded = PCA_encoding(forecast.copy().drop(columns=['o_road']), direction)
    forecast = PCA_encoded.join(forecast[['o_road']])
    one_hot_encoder = get_one_hot_encoder(forecast)
    forecast = one_hot_encoding(forecast.copy(), one_hot_encoder)   
    OH_forecast = forecast.copy().drop(columns=[col for col in forecast.columns if col.startswith('PC')])

    return forecast, OH_forecast

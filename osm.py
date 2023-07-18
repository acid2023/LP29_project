import re
import requests
import logging
import pickle
from typing import Tuple
from shapely.geometry import Point
from shapely.ops import unary_union
import geopandas as gpd
import pandas as pd

import folders
from folders import folder_check

coordinates_filename = "coordinates_dict.pkl"
station_coords = {}


def load_dicts() -> None:
    global dict_locations, dict_ops_id, roads_areas
    path = folder_check(folders.dict_folder)
    with open(f'{path}dict_locations.pkl', 'rb') as f:
        dict_locations = pickle.load(f)
    with open(f'{path}dict_ops_id.pkl', 'rb') as f:
        dict_ops_id = pickle.load(f)
    with open(f'{path}roads_areas.pkl', 'rb') as f:
        roads_areas = pickle.load(f)


def save_dicts() -> None:
    global dict_locations, dict_ops_id, roads_areas
    path = folder_check(folders.dict_folder)
    with open(f'{path}dict_locations.pkl', 'wb') as f:
        pickle.dump(dict_locations, f)
    with open(f'{path}dict_ops_id.pkl', 'wb') as f:
        pickle.dump(dict_ops_id, f)
    with open(f'{path}roads_areas.pkl', 'wb') as f:
        pickle.dump(roads_areas, f)


def save_coordinates_dict() -> None:
    path = folder_check(folders.dict_folder)
    filename = f'{path}{coordinates_filename}'
    global station_coords
    with open(filename, "wb") as f:
        pickle.dump(station_coords, f)


def load_coordinates_dict() -> None:
    global station_coords
    path = folder_check(folders.dict_folder)
    filename = f'{path}{coordinates_filename}'
    try:
        with open(filename, "rb") as f:
            station_coords = pickle.load(f)
    except FileNotFoundError:
        logging.warning(f"{filename} not found, creating new empty dictionary")
        station_coords = {}


def get_code(location):
    pattern = r'\((\d+)\)[^(]*$'
    if not isinstance(location, str):
        location = str(location)
    match = re.search(pattern, location)
    if match:
        return match.group(1)
    else:
        return None


def fetch_coords_from_dicts(station: str) -> Tuple[float, float]:
    global dict_locations, dict_ops_id

    ops_id = get_code(station)

    location = station.split(' ')[0].upper().rstrip()
    if ops_id:
        coords = dict_ops_id.get(ops_id, 0)
        ops_id_2 = ops_id[:-1]
        if coords:
            return coords
        else:
            coords = dict_ops_id.get(ops_id_2, 0)
            if coords:
                return coords

    coords_2 = dict_locations.get(location, 0)
    if coords_2:
        return coords_2

    return [None, None]


def fetch_coordinates(station: str) -> Tuple[float, float]:
    global station_coords
    url = 'https://nominatim.openstreetmap.org/search?'
    pattern = r'\([^)]*\)'

    if not isinstance(station, str):
        station = str(station)

    results = fetch_coords_from_dicts(station)

    if results != [None, None]:
        return results

    if station in station_coords:
        results = station_coords[station]
        return results

    try:
        location = re.sub(pattern, "", station).strip()
        params = {'q': location, 'format': 'json', 'railway': 'station, stop, halt'}
        response = requests.get(f"{url}{'&'.join([f'{k}={v}' for k, v in params.items()])}")
        results = response.json()

        try:
            coords = [results[0]['lat'], results[0]['lon']]
            if coords:
                logging.error(f'coordinates for {station} found')
                station_coords[station] = coords
                return coords
            else:
                logging.error('problems parsing geodata')
                return [0, 0]
        except Exception as e:
            logging.exception(f'{station}problems %s', e)
            return [0, 0]
    except Exception as e:
        logging.exception(f'{station}problems %s', e)
        return [0, 0]


def road_check(coords, road):
    if not road:
        return False
    global roads_areas
    area = roads_areas.get(road, None)
    if area is None:
        return False
    lat, lon = coords
    if not coords or (lat is None or lon is None):
        return False
    return area.contains(Point(lon, lat))


def update_roads_areas(df: pd.DataFrame) -> None:
    global roads_areas
    geometry = []
    for row in df.itertuples():
        lat = getattr(row, 'lat')
        lon = getattr(row, 'lon')
        geometry.append(Point(lon, lat))
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf['buffer'] = gdf['geometry'].buffer(1)
    for row in gdf.itertuples(index=False):
        area = getattr(row, 'o_road')
        buffer_zone = getattr(row, 'buffer')
        if area in roads_areas:
            roads_areas[area] = unary_union([roads_areas[area], buffer_zone])
        else:
            roads_areas[area] = buffer_zone
    save_dicts()


def update_coordinates_dict(df: pd.DataFrame) -> None:
    global station_coords, dict_ops_id
    for row in df.itertuples(index=False):
        station = getattr(row, 'ops_station')
        ops_id = get_code(station)
        lat = getattr(row, 'lat')
        lon = getattr(row, 'lon')
        station_coords[station] = [lat, lon]
        dict_ops_id[ops_id] = [lat, lon]
    save_coordinates_dict()
    save_dicts()


load_coordinates_dict()
load_dicts()

if __name__ == "__main__":
    pass

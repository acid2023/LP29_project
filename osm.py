import re
import requests
import logging
import pickle
from typing import Tuple

import folders
from folders import folder_check

coordinates_filename = "coordinates_dict.pkl"
station_coords = {}


def load_dicts() -> None:
    global dict_locations, dict_ops_id
    path = folder_check(folders.dict_folder)
    with open(f'{path}dict_locations.pkl', 'rb') as f:
        dict_locations = pickle.load(f)
    with open(f'{path}dict_ops_id.pkl', 'rb') as f:
        dict_ops_id = pickle.load(f)


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


def fetch_coords_from_dicts(station: str) -> Tuple[float, float]:
    def get_code(location):
        pattern = r'\((\d+)\)[^(]*$'
        if not isinstance(location, str):
            location = str(location)
        match = re.search(pattern, location)
        if match:
            return match.group(1)
        else:
            return None

    global dict_locations, dict_ops_id

    ops_id = get_code(station)

    location = station.split(' ')[0].upper()[:-1]
    if ops_id:
        coords = dict_ops_id.get(ops_id, None)
        ops_id_2 = ops_id[:-1]
        if coords:
            return coords
        else:
            coords = dict_ops_id.get(ops_id_2, None)
            if coords:
                return coords

    coords_2 = dict_locations.get(location, None)
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

    if results:
        return results

    if station in station_coords:
        return station_coords[station]

    try:
        location = re.sub(pattern, "", station).strip()
        params = {'q': location, 'format': 'json', 'railway': 'station' | 'stop' | 'halt'}
        response = requests.get(f"{url}{'&'.join([f'{k}={v}' for k, v in params.items()])}")
        results = response.json()

        try:
            coords = [results[0]['lat'], results[0]['lon']]
            if coords:
                logging.info(f'coordinates for {station} found')
                station_coords[station] = coords
                return coords
            else:
                logging.info('problems parsing geodata')
        except Exception as e:
            logging.exception('problems %s', e)
            return [None, None]
    except Exception as e:
        logging.exception('problems %s', e)
        return [None, None]


load_coordinates_dict()
load_dicts()

if __name__ == "__main__":
    pass

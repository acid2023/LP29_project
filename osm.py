import re
import requests
import logging
import pickle
import folders

coordinates_filename = "coordinates_dict.pkl"
station_coords = {}


def fetch_coordinates(station: str) -> list:
    global station_coords
    if not isinstance(station, str):
        station = str(station)
    if station in station_coords:
        # logging.info(f'coordinates for {station} found in dictionary')
        return station_coords[station]
    else:
        try:
            url = 'https://nominatim.openstreetmap.org/search?'
            pattern = r'\([^)]*\)'
            location = re.sub(pattern, "", station).strip()
            params = {'q': location, 'format': 'json'}
            response = requests.get(f"{url}{'&'.join([f'{k}={v}' for k, v in params.items()])}")
            results = response.json()
            location_2 = 'железнодорожная станция ' + location
            params_2 = {'q': location_2, 'format': 'json'}
            response_2 = requests.get(f"{url}{'&'.join([f'{k}={v}' for k, v in params_2.items()])}")
            results_2 = response_2.json()
            try:
                coords = [results[0]['lat'], results[0]['lon']]
                coords_2 = [results_2[0]['lat'], results_2[0]['lon']]
                if coords_2 != [None, None]:
                    coords = coords_2
                if coords:
                    logging.info(f'coordinates for {station} found')
                    station_coords[station] = coords
                    return coords
                else:
                    logging.info('problems parsing geodata')
            except:
                pass
                return [None, None]
        except Exception as e:
            logging.exception('problems %s', e)
            return [None, None]


def save_coordinates_dict() -> None:
    path = folders.folder_check(folders.models_folder)
    filename = f'{path}{coordinates_filename}'
    global station_coords
    with open(filename, "wb") as f:
        pickle.dump(station_coords, f)


def load_coordinates_dict() -> None:
    global station_coords
    path = folders.folder_check(folders.models_folder)
    filename = f'{path}{coordinates_filename}'
    try:
        with open(filename, "rb") as f:
            station_coords = pickle.load(f)
    except FileNotFoundError:
        logging.warning(f"{filename} not found, creating new empty dictionary")
        station_coords = {}


load_coordinates_dict()


if __name__ == "__main__":
    pass

import pandas as pd
import tensorflow as tf

import telegram
from telegram.error import TelegramError

import io
import sys
import os
import time
import logging
import email
# import select

from mail import get_data_from_message, get_messages, send_letter, archiveing_and_removing_messages
import mail_settings as ms
from authorize import authorize_user, generate_new_user_signature
from folders import folder_check
import folders
import modeling as md
import modeling_settings as mds
from bot import TelegramBotHandler
import osm


def start_logging(**kwarg: str | TelegramBotHandler) -> str:
    logging.basicConfig(level=logging.INFO)
    screen = kwarg.get('screen', None)
    bot = kwarg.get('bot', None)
    path = folder_check(folders.logs_folder)
    unix_time = int(time.time())
    filename = f'{path}{unix_time}_{ms.log_file}'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if screen:
        if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
            screen_handler = logging.StreamHandler()
            screen_handler.setLevel(logging.INFO)
            screen_handler.setFormatter(formatter)
            logger.addHandler(screen_handler)
    if bot:
        try:
            bot_handler = TelegramBotHandler(bot)
            bot_handler.setLevel(logging.ERROR)
            bot_handler.setFormatter(formatter)
            logger.addHandler(bot_handler)
        except TelegramError as e:
            logging.ERROR('unable to initiate TelegramBot: %s' % e)
    return filename


def df_to_excel(df: pd.DataFrame) -> bytes:
    with io.BytesIO() as buffer:
        df.to_excel(buffer, index=False)
        return buffer.getvalue()


def load_dataframe(filename):
    pickle_file = filename + '.pkl'
    excel_file = filename + '.xlsx'
    if os.path.isfile(pickle_file):
        df = pd.read_pickle(pickle_file)
    else:
        df = pd.read_excel(excel_file)
        df.to_pickle(pickle_file)
    return df


def check_coordinates(df: pd.DataFrame):
    df.dropna(subset=['ops station', 'o_road'], inplace=True)
    df.reset_index(drop=True)
    unique_stations = df[~df.duplicated(subset=['ops station'])]
    station_check = {}
    for idx, row in unique_stations.iterrows():
        ops_station = row['ops station']
        coords = osm.fetch_coordinates(ops_station)
        o_road = row['o_road']
        station_check[idx] = (ops_station, o_road, coords[0], coords[1], osm.road_check(coords, o_road))
    station_check = pd.DataFrame(station_check).T
    station_check.columns = ['ops_station', 'o_road', 'lat', 'lon', 'check']
    return station_check[station_check.check == False]


def check_geodata(df: pd.DataFrame, **kwargs) -> None:
    letter = kwargs.get('letter', False)
    coords_check = check_coordinates(df)
    problem_stations = df_to_excel(coords_check)
    if not coords_check.empty:
        logging.error('there are problems with geodata for stations - wrong geodata parsed')
        if letter:
            send_letter(letter['sender'], 'problem stations', message_type='xlsx',
                        attachment=problem_stations, filename='problem_stations.xlsx')
            logging.error('problem stations sent')
        else:
            coords_check.to_excel('problem_stations.xlsx')
            logging.error('problem stations saved')
    else:
        logging.error('no problems with parsed geodata for stations found')


def create_models(**kwargs: str | email.message.Message) -> None:
    logging.error('creating models')
    letter = kwargs.get('letter', False)
    filename = kwargs.get('filename', False)
    if not filename:
        filename = 'TH_0105_0507'
    local = kwargs.get('local', False)
    if letter:
        attachment = get_data_from_message(letter['message'], get_type='xlsx')
        logging.error('attachement found')
        xlsx_data = attachment[0]['data']
        xlsx_filename = attachment[0]['filename']
        df = pd.read_excel(xlsx_data)
        logging.error(f'loaded attachment - {xlsx_filename}')
        df.to_excel(xlsx_filename)
        logging.error('attachment saved local for future use')
    elif local:
        logging.error('loading local data')
        df = load_dataframe(filename)
        logging.error('local data loaded')
    else:
        return
    check_geodata(df, leter=letter)
    # created_models_dict = md.create_models(df, mds.DefaultColumns)
    md.create_2_way_models(df)
    logging.error('models created')
    # md.save_models(created_models_dict)
    logging.error('no errors found, models saved')


def predict_data(**kwargs: str | email.message.Message) -> None:
    letter = kwargs.get('letter', False)
    local = kwargs.get('local', False)
    filename = kwargs.get('filename', False)
    logging.error('predicting data')
    if letter:
        attachment = get_data_from_message(letter['message'], get_type='xlsx')
        logging.error('attachement found')
        xlsx_data = attachment[0]['data']
        df = pd.read_excel(xlsx_data)
        logging.error('loaded attachment')
    elif local:
        logging.error('loading local data')
        df = load_dataframe(filename)
        logging.error('local data loaded')
    else:
        return
    check_geodata(df, letter=letter)
    forecast = md.prediction(df)
    logging.error('forecast completed')
    update_trains = df_to_excel(forecast)
    logging.error('updated data on trains compiled')
    if letter:
        send_letter(letter['sender'], 'forecasted data', message_type='xlsx',
                    attachment=update_trains, filename='update_trains_new.xlsx')
        logging.info('updates sent, no error found')
    if letter or local:
        forecast.to_excel('update_trains_new.xlsx')
        logging.error('update saved locally')


def geodata_update(**kwargs: str | email.message.Message) -> None:
    letter = kwargs.get('letter', False)
    local = kwargs.get('local', False)
    filename = kwargs.get('filename', False)
    logging.error('updating geodata')
    if letter:
        attachment = get_data_from_message(letter['message'], get_type='xlsx')
        logging.error('attachement found')
        xlsx_data = attachment[0]['data']
        df = pd.read_excel(xlsx_data)
        logging.error('loaded attachment')
    elif local:
        logging.error('loading local data')
        df = load_dataframe(filename)
        logging.error('local data loaded')
    try:

        osm.update_coordinates_dict(df)
        osm.update_roads_areas(df)
    except Exception as e:
        logging.exception('error updating geodata: %s', e)
    logging.error('geodata updated')


def main(local_mode: bool, filename: str | bool, local_choice: str | bool) -> None:
    if not local_mode:
        emails = get_messages(all_messages=True)
        logging.error(f'recieved {len(emails)} letters')
        archive_list = []
        for letter_num, letter in enumerate(emails):
            logging.info(f'authorizing user from letter {letter_num}')
            user_master = (letter['sender'] == ms.master_mail or letter['return_path'] == ms.master_mail)
            user_athorized = authorize_user(letter) or user_master
            logging.info(f"user {letter['sender']} authorized: {user_athorized}")
            if user_athorized:
                logging.info('authorizing user done')
                archive_list.append(letter['message_id'])
            else:
                logging.info(f"user {letter['sender']} is not authorized")
                continue
            if user_master and letter['subject'] == ms.new_user_subject:
                logging.info('new user signature creating')
                signature = generate_new_user_signature(letter)
                logging.info('signature for new user created')
                text = f'signature="{signature}"'
                send_letter(letter['sender'], 'Access to models server', message_type='message', message=text)
                logging.info('new user signature sent')
            elif user_athorized and letter['subject'] == ms.creation_subject:
                create_models(letter=letter)
            elif user_athorized and letter['subject'] == ms.prediction_subject:
                predict_data(letter=letter)
            elif user_athorized and letter['subject'] == ms.geodata_update_subject:
                geodata_update(letter=letter)
        if archive_list:
            logging.error('archiving messages from authorized users')
            archiveing_and_removing_messages(archive_list)
    else:
        request = ('select action: (1) create/ (2)predict/ (3) validation test/ '
                   '(4) post modeling validation/ (5) update geopdata/ (6) exit: ')
        if not local_choice:
            select_action = input(request)
        else:
            select_action = local_choice
        if select_action == '1':
            logging.info('local_mode - creating models')
            create_models(local=True, filename=filename)
        elif select_action == '2':
            logging.info('local mode - predicting models')
            predict_data(local=True, filename=filename)
        elif select_action == '3':
            logging.info('local mode - validation test')
            df = load_dataframe(filename)
            md.validate_models(df)
        elif select_action == '4':
            logging.info('local mode - post modeling validation test')
            df = load_dataframe(filename)
            md.validating_on_post_data(df)
        elif select_action == '5':
            geodata_update(local=True, filename=filename)
        elif select_action == '6':
            exit(0)


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print('GPU is available')
        # Set the GPU as the default device
        device = gpus[0]
        tf.config.experimental.set_memory_growth(device, False)
        tf.config.experimental.set_virtual_device_configuration(
           device,
           [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0.75)])
        # tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print('GPU is not available')
    arguments = len(sys.argv)
    local_mode = arguments > 1 and sys.argv[1] == 'local'
    if not local_mode:
        try:
            my_bot = telegram.Bot(token=ms.API_KEY)
            logs_file = start_logging(screen=True, bot=my_bot)
        except TelegramError as e:
            logs_file = start_logging(screen=True)
            logging.error('start')
            logging.error('unable to initiate TelegramBot: %s' % e)
    else:
        logs_file = start_logging(screen=True)
        logging.error('start')
    filename = False
    local_choice = False
    if arguments >= 3:
        filename = sys.argv[2]
    if arguments >= 4:
        local_choice = sys.argv[3]
    if local_mode:
        logging.error('Local mode - saved files will be used for creating or predicting')
    try:
        while True:
            main(local_mode, filename, local_choice)
            start_time = time.time()
            next_iteration = False
            '''
            print('press any key or it restart loop in 10 minutes')
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 60)
                if ready:
                    break
            elapsed_time = time.time() - start_time
            if elapsed_time >= 60:
                next_iteration = True
                break
            '''
            if not next_iteration:
                if input('stop: (y/n)') == 'y':
                    break
            local_choice = False
    except (ValueError, AttributeError, TypeError, KeyError, IndexError) as e:
        logging.exception('error found: %s', e)
    except Exception as e:
        logging.exception('unknown error found: %s', e)
    finally:
        try:
            if not local_mode:
                send_letter(ms.master_mail, 'logs', message_type='logs', filename=logs_file)
            logging.info('logs sent')
        except Exception as e:
            logging.exception('error found: %s', e)
            logging.info('logs not sent')

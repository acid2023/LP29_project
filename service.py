import pandas as pd

import telegram
from telegram.error import TelegramError

import io
import sys
import time
import logging
import email

from mail import get_data_from_message, get_messages, send_letter
import mail_settings as ms
from authorize import authorize_user, generate_new_user_signature
from folders import folder_check
import folders
import modeling as md
import modeling_settings as mds
from bot import TelegramBotHandler


def start_logging(**kwarg: str | TelegramBotHandler) -> str:
    logging.basicConfig(level=logging.DEBUG)
    screen = kwarg.get('screen', None)
    bot = kwarg.get('bot', None)
    path = folder_check(folders.logs_folder)
    unix_time = int(time.time())
    filename = f'{path}{unix_time}_{ms.log_file}'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s - Line No:%(lineno)d - %(message)s')
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
            bot_handler.setLevel(logging.INFO)
            bot_handler.setFormatter(formatter)
            logger.addHandler(bot_handler)
        except TelegramError as e:
            logging.ERROR('unable to initiate TelegramBot: %s' % e)
    return filename


def df_to_excel(df: pd.DataFrame) -> bytes:
    with io.BytesIO() as buffer:
        df.to_excel(buffer, index=False)
        return buffer.getvalue()


def create_models(**kwargs: str | email.Message) -> None:
    logging.info('creating models')
    letter = kwargs.get('letter', False)
    local = kwargs.get('local', False)
    if letter:
        attachment = get_data_from_message(letter['message'], get_type='xlsx')
        logging.info('attachement found')
        xlsx_data = attachment[0]['data']
        xlsx_filename = attachment[0]['filename']
        df = pd.read_excel(xlsx_data)
        logging.info(f'loaded attachment - {xlsx_filename}')
    elif local:
        logging.info('loading local data')
        df = pd.read_excel('2Home_data.xlsx')
        logging.info('local data loaded')
    else:
        return
    created_models_dict = md.create_models(df, mds.DefaultColumns)
    logging.info('models created')
    md.save_models(created_models_dict)
    logging.info('models saved')
    logging.info('no errors found, models saved')


def predict_data(**kwargs: str | email.Message) -> None:
    letter = kwargs.get('letter', False)
    local = kwargs.get('local', False)
    logging.info('predicting data')
    if letter:
        attachment = get_data_from_message(letter['message'], get_type='xlsx')
        logging.info('attachement found')
        xlsx_data = attachment[0]['data']
        df = pd.read_excel(xlsx_data)
        logging.info('loaded attachment')
    elif local:
        logging.info('loading local data')
        df = pd.read_excel('TH_0105_1306.xlsx')
        logging.info('local data loaded')
    else:
        return
    logging.info('Loading models')
    models_dict = md.load_models()
    logging.info('Models loaded')
    forecast = md.prediction(df, models_dict)
    logging.info('forecast completed')
    update_trains = df_to_excel(forecast)
    logging.info('updated data on trains compiled')
    if letter:
        send_letter(letter['sender'], 'forecasted data', message_type='xlsx',
                    attachment=update_trains, filename='update_trains_new.xlsx')
        logging.info('updates sent, no error found')
    if letter or local:
        forecast.to_excel('update_trains.xlsx')
        logging.info('update saved locally')


def main(local_mode: bool) -> None:
    if not local_mode:
        emails = get_messages(all_messages=True)
        logging.info(f'recieved {len(emails)} letters')
        archive_list = []
        for letter_num, letter in enumerate(emails):
            logging.info(f'authorizing user from letter {letter_num}')
            user_master = (letter['sender'] == ms.master_mail or letter['return_path'] == ms.master_mail)
            logging.info(f'user master: {user_master}')
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
                create_models(letter)
            elif user_athorized and letter['subject'] == ms.prediction_subject:
                predict_data(letter)
    else:
        select_action = input('select creating or predicting: (1) create/ (2)predict: ')
        if select_action == '1':
            create_models(local=True)
            logging.info('local_mode - creating models')
        elif select_action == '2':
            predict_data(local=True)
            logging.info('local mode - predicting models')
    # archiveing_and_removing_messages()


if __name__ == "__main__":
    my_bot = telegram.Bot(token=ms.API_KEY)
    logs_file = start_logging(screen=True)
    logging.info('start')
    local_mode = len(sys.argv) > 1 and sys.argv[1] == 'local'
    if local_mode:
        logging.info('Local mode - saved files will be used for creating or predicting')
    try:
        while True:
            main(local_mode)
            local_mode = input('continue locally - (y/n)') == 'y'
            if input('stop: (y/n)') == 'y':
                break
    except (ValueError, AttributeError, TypeError, KeyError, IndexError) as e:
        logging.exception('error found: %s', e)
    except Exception as e:
        logging.exception('unknown error found: %s', e)
    finally:
        try:
            send_letter(ms.master_mail, 'logs', message_type='logs', filename=logs_file)
            logging.info('logs sent')
        except Exception as e:
            logging.exception('error found: %s', e)
            logging.info('logs not sent')

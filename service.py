import pandas as pd

import telegram
from telegram.error import TelegramError

import io
import sys
import os
import time
import logging
import email

from mail import get_data_from_message, get_messages, send_letter, archiveing_and_removing_messages
import mail_settings as ms
from authorize import authorize_user, generate_new_user_signature
from folders import folder_check
import folders
import modeling as md
import modeling_settings as mds
from bot import TelegramBotHandler


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


def create_models(**kwargs: str | email.message.Message) -> None:
    logging.error('creating models')
    letter = kwargs.get('letter', False)
    filename = kwargs.get('filename', False)
    if not filename:
        filename = 'TH_0105_2006'
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
    created_models_dict = md.create_models(df, mds.DefaultColumns)
    logging.error('models created')
    md.save_models(created_models_dict)
    logging.error('no errors found, models saved')


def predict_data(**kwargs: str | email.message.Message) -> None:
    letter = kwargs.get('letter', False)
    local = kwargs.get('local', False)
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
    forecast = md.prediction(df)
    logging.error('forecast completed')
    update_trains = df_to_excel(forecast)
    logging.error('updated data on trains compiled')
    if letter:
        send_letter(letter['sender'], 'forecasted data', message_type='xlsx',
                    attachment=update_trains, filename='update_trains_new.xlsx')
        logging.info('updates sent, no error found')
    if letter or local:
        forecast.to_excel('update_trains.xlsx')
        logging.error('update saved locally')


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
        if archive_list:
            logging.errr('archiving messages from authorized users')
            archiveing_and_removing_messages(archive_list)
    else:
        request = 'select action: (1) create/ (2)predict/ (3) validation test/ (4) post modeling validation/ (5) KeraTune/ (6) exit: '
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
            logging.info('local mode - KeraTuning model')
            md.KeraTune()
        elif select_action == '6':
            exit(0)


if __name__ == "__main__":
    my_bot = telegram.Bot(token=ms.API_KEY)
    logs_file = start_logging(screen=True)
    logging.error('start')
    arguments = len(sys.argv)
    local_mode = arguments > 1 and sys.argv[1] == 'local'
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
            if input('stop: (y/n)') == 'y':
                break
            local_choice = False
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

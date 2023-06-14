from mail import load_xlsx_from_message, get_columns_list_from_message, send_logs, send_xlsx, get_messages, send_message
import mail_settings as ms
from authorize import authorize_user, generate_new_user_signature
from folders import folder_check
import folders
import time
import logging
import pandas as pd
import modeling as md
import modeling_settings as mds
import io


def start_logging():
    path = folder_check(folders.logs_folder)
    unix_time = int(time.time())
    filename = f'{path}{unix_time}_{ms.log_file}'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return filename


def df_to_excel(df: pd.DataFrame) -> bytes:
    with io.BytesIO() as buffer:
        df.to_excel(buffer, index=False)
        return buffer.getvalue()


def create_models(letter: bytes):
    logging.info('creating models')
    attachment = load_xlsx_from_message(letter['message'])
    logging.info('attachement found')
    xlsx_data = attachment[0]['data']
    xlsx_filename = attachment[0]['filename']
    df = pd.read_excel(xlsx_data)
    df.drop(df[df['update'] >= pd.to_datetime(mds.DefaultTrainingDateCut)].index, inplace=True)
    logging.info(f'loaded attachment - {xlsx_filename}')
    columns_list = get_columns_list_from_message(letter)
    models = md.create_models(df, columns_list)
    logging.info('models created')
    md.save_models(models)
    logging.info('models saved')
    logging.info('no errors found, models saved')


def predict_data(letter: bytes):
    logging.info('predicting data')
    attachment = load_xlsx_from_message(letter['message'])
    logging.info('attachement found')
    xlsx_data = attachment[0]['data']
    df = pd.read_excel(xlsx_data)
    df.drop(df[df['update'] <= pd.to_datetime(mds.DefaultTrainingDateCut)].index, inplace=True)
    columns_list = get_columns_list_from_message(letter)
    logging.info('Loading models')
    models = md.load_models()
    logging.info('Models loaded')
    forecast = md.prediction(df, models, columns_list)
    logging.info('forecast completed')
    update_trains = df_to_excel(forecast)
    logging.info('updated data on trains compiled')
    send_xlsx(letter['sender'], 'updated data on trains', update_trains, 'update_trains.xlsx')
    logging.info('updates sent, no error found')
    forecast.to_excel('update_trains.xlsx')
    logging.info('update saved locally')


def main():
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
            send_message(letter['sender'], 'Access to models server', f'signature="{signature}"')
            logging.info('new user signature sent')
        elif user_athorized and letter['subject'] == ms.creation_subject:
            create_models(letter)
        elif user_athorized and letter['subject'] == ms.prediction_subject:
            predict_data(letter)
    # archiveing_and_removing_messages()


if __name__ == "__main__":
    logs_file = start_logging()
    logging.info('start')
    try:
        while True:
            main()
            if input('stop: (y/n)') == 'y':
                break
    except ValueError as e:
        logging.info('error found: %', e)
    except AttributeError as e:
        logging.info('error found: %', e)
    except TypeError as e:
        logging.info('error found: %', e)
    except KeyError as e:
        logging.info('error found: %', e)
    except IndexError as e:
        logging.info('error found: %', e)
    except Exception as e:
        logging.info('error found: %', e)
    finally:
        try:
            send_logs(ms.master_mail, logs_file)
            logging.info('logs sent')
        except Exception as e:
            logging.info('error found: %', e)
            logging.info('logs not sent')

import email
import imaplib
from email.header import decode_header
import re
import mail_settings as ms
from authorize import authorize_user, generate_new_user_signature
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
import io
import modeling as md
import logging
import os
import time
import modeling_settings as mds


def imap_login():
    username = ms.username
    mail_pass = ms.mail_pass
    imap_server = ms.imap_server
    imap = imaplib.IMAP4_SSL(imap_server)
    imap.login(username, mail_pass)
    imap.select("INBOX")
    return imap


def stmp_login():
    username = ms.username
    mail_pass = ms.mail_pass
    smtp = smtplib.SMTP_SSL(ms.smtp_server)
    smtp.login(username, mail_pass)
    return smtp


def message_extract(imap, request):
    _, response = imap.fetch(request, '(RFC822)')
    msg = email.message_from_bytes(response[0][1])
    return msg


def get_message_by_id(message_id):
    return message_extract(imap_login(), message_id)

def decode(bytes):
    try:
        return decode_header(bytes)[0][0].decode()
    except AttributeError:
        return bytes


def get_messages(all_messages=False):
    email_pattern = r'<([^<>]+)>'
    mail = imap_login()
    mail.select('INBOX')
    emails_list = []
    search_querry = 'ALL' if all_messages else 'UNSEEN'
    _, data = mail.search(None, search_querry)
    message_nums = data[0].split()
    i = 0
    for message_num in message_nums:
        msg = message_extract(mail, message_num)
        message_id = msg['Message-ID']
        subject = msg['Subject']
        subject = decode(subject)
        logging.info(f'Message {i}, subject - {subject}')
        sender = re.search(email_pattern, msg['From']).group(1)
        msg_date = msg['Date']
        return_path = msg['Return-path']
        emails_list.append({'message_id': message_id,
                            'message': msg,
                            'subject': subject, 'date': msg_date,
                            'sender': sender, 'return_path': return_path})
        user_master = sender == ms.master_mail or return_path == ms.master_mail
        user_athorized = authorize_user(emails_list[-1]) or user_master
        if user_athorized:
            #mail.select('ARCHIVE')
            mail.copy(message_num, "ARCHIVE")
            logging.info(f"message from authorized user {sender} copyied to ARCHIVE")
        else:
            #mail.select('TRASH')
            mail.copy(message_num, "TRASH")
            logging.info(f"message from unauthorized user {sender} copyied to TRASH")
        i += 1
    return emails_list


def remove_message():
    logging.info("removing message")
    mail = imap_login()
    _, data = mail.search(None, 'ALL')
    message_nums = data[0].split()
    for message_num in message_nums:
        mail.store(message_num, "+FLAGS", '\\Deleted')
    
        

def load_xlsx_from_message(message):
    xlsx_files = []
    for id_part, part in enumerate(message.walk()):
        try:
            filename = decode_header(part.get_filename())[0][0].decode() if part.get_filename() else ""
        except AttributeError:
            filename = part.get_filename() if part.get_filename() else ""
        if filename.endswith('.xlsx'):
            xlsx_data = part.get_payload(decode=True)
            xlsx_files.append({'filename': filename, 'data': xlsx_data})
    return xlsx_files


def get_text_body_from_message(message):
    text_parts = []
    for part in message.walk():
        if part.get_content_type() == 'text/plain':
            text_parts.append(part.get_payload(decode=True).decode('utf-8'))
    text = '\n'.join(text_parts)
    return text


def get_columns_list_from_message(letter):
    default_columns = mds.DefaultColumns
    message = letter['message']
    letter_test = get_text_body_from_message(message)
    pattern = r'list of columns: \[(.*?)\]'
    match = re.search(pattern, letter_test)
    if match:
        column_str = match.group(1)
        columns = [col.strip() for col in column_str.split(',')]
        return columns
    else:
        return default_columns


def send_attachment(receiver_address, subject, attachment):
    mail = stmp_login()
    my_address = ms.username
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = my_address
    msg['To'] = receiver_address
    msg.attach(attachment)
    mail.sendmail(my_address, receiver_address, msg.as_bytes())


def send_xlsx(receiver_address, subject, data, filename):
    attachment = MIMEApplication(data, _subtype='xlsx')
    attachment['Content-Disposition'] = f'attachment; filename="{filename}"'
    send_attachment(receiver_address, subject, attachment)


def send_logs(receiver_address, logs_file):
    with open(logs_file, 'rb') as file:
        session_log = file.read()
    attachment = MIMEApplication(session_log, _subtype='plain')
    attachment['Content-Disposition'] = f'attachment; filename={logs_file}'
    send_attachment(receiver_address, 'Logs', attachment)


def send_message(receiver_address, subject, message):
    mail = stmp_login()
    my_address = ms.username
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = my_address
    msg['To'] = receiver_address
    msg.attach(MIMEText(message, 'plain'))
    mail.sendmail(my_address, receiver_address, msg.as_string())


def df_to_excel(df):
    with io.BytesIO() as buffer:
        df.to_excel(buffer, index=False)
        return buffer.getvalue()


def start_logging():
    unix_time = int(time.time())
    filename = f'{unix_time}_{ms.log_file}'
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
    

def create_models(letter):
    logging.info('creating models')
    attachment = load_xlsx_from_message(letter['message'])
    logging.info('attachement found')
    xlsx_data = attachment[0]['data']
    xlsx_filename = attachment[0]['filename']
    df = pd.read_excel(xlsx_data)
    df.drop(df[df['update'] >= pd.to_datetime('2023-05-01')].index, inplace=True)
    logging.info(f'loaded attachment - {xlsx_filename}')
    columns_list = get_columns_list_from_message(letter)
    models = md.create_models(df, columns_list)
    logging.info('models created')
    md.save_models(models)
    logging.info('models saved')
    logging.info('no errors found, models saved')


def predict_data(letter):
    logging.info('predicting data')
    attachment = load_xlsx_from_message(letter['message'])
    logging.info('attachement found')
    xlsx_data = attachment[0]['data']
    df = pd.read_excel(xlsx_data)
    df.drop(df[df['update'] < pd.to_datetime('2023-05-01')].index, inplace=True)
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


def mail_folders_setup():
    mail = imap_login()
    logging.info('setting up folders in INBOX')
    # i need to check if folder 'archive' and 'trash' exist, if not create them
    status_exist, response = mail.select('ARCHIVE')
    if status_exist != 'OK':
        mail.create('ARCHIVE')
        logging.info('creating ARCHIVE folder')
    else:
        logging.info('folder ARCHIVE already exists')
    status_exist, response = mail.select('TRASH')
    if status_exist != 'OK':
        logging.info('creating TRASH folder')
        mail.create('TRASH')
    else:
        logging.info('folder TRASH already exist')

def main():
    mail_folders_setup()
    emails = get_messages(all_messages=True)
    logging.info(f'recieved {len(emails)} letters')
    for letter_num, letter in enumerate(emails):
        logging.info(f'authorizing user from letter {letter_num}')
        user_master = (letter['sender'] == ms.master_mail or letter['return_path'] == ms.master_mail)
        logging.info(f'user master: {user_master}')
        user_athorized = authorize_user(letter) or user_master
        logging.info(f"user {letter['sender']} authorized: {user_athorized}")
        if user_athorized:
            logging.info('authorizing user done')
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
    remove_message()


if __name__ == "__main__":
    logs_file = start_logging()
    logging.info('start')
    try:
        main()
    except Exception as e:
        logging.exception('error found: %', e)
    finally:
        try:
            send_logs(ms.master_mail, logs_file)
            logging.info('logs sent')
        except Exception as e:
            logging.exception('error found: %', e)
            logging.info('logs not sent')

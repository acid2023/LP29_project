import email
import imaplib
from email.header import decode_header
import base64
import re
import mail_settings as ms
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email import encoders
import io
import modeling as md
import logging
import os


def imap_login():
    username = ms.username
    mail_pass = ms.mail_pass
    imap = imaplib.IMAP4_SSL(ms.imap_server)
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


def get_messages(all_messages=False):
    email_pattern = r'<([^<>]+)>'
    mail = imap_login()
    emails_list = []
    search_querry = 'ALL' if all_messages else 'UNSEEN'
    _, data = mail.search(None, search_querry)
    message_nums = data[0].split()
    for message_num in message_nums:
        msg = message_extract(mail, message_num)
        message_id = msg['Message-ID']
        subject = msg['Subject']
        subject = decode_header(subject)[0][0].decode()
        sender = msg['From']
        sender = re.search(email_pattern, sender).group(1)
        msg_date = msg['Date']
        return_path = msg['Return-path']
        emails_list.append({'message_id': message_id,
                            'message': msg,
                            'subject': subject, 'date': msg_date,
                            'sender': sender, 'return_path': return_path})
    return emails_list


def load_xlsx_from_message(message):
    xlsx_files = []
    for id_part, part in enumerate(message.walk()):
        filename = part.get_filename()
        if filename:
            filename = decode_header(filename)[0][0].decode()
            if filename.find('.xlsx') != -1:
                xlsx_data = part.get_payload(decode=True)
                xlsx_files.append({'filename': filename, 'data': xlsx_data})
    return xlsx_files


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


def send_logs(receiver_address):
    logs_file = ms.log_file
    with open(logs_file, 'rb') as file:
        session_log = file.read()
    attachment = MIMEApplication(session_log, _subtype='plain')
    attachment['Content-Disposition'] = f'attachment; filename={logs_file}'
    send_attachment(receiver_address, 'Logs', attachment)


def df_to_excel(df):
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()    


def start_logging():
    logs_file = ms.log_file
    if os.path.exists(logs_file):
        os.remove(logs_file)
    logging.basicConfig(filename=logs_file, format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)


def main():
    start_logging()
    logging.info('start')
    print('logging on')
    emails = get_messages(all_messages=True)
    print(f'recieved {len(emails)} letters')
    logging.info(f'recieved {len(emails)} letters')
    for letter in emails:
        if letter['sender'] == ms.master_mail and letter['subject'] == ms.creation_subject:
            print('creating models')
            logging.info('creating models')
            try:
                attachment = load_xlsx_from_message(letter['message'])
                print('attachement found')
                xlsx_data = attachment[0]['data']
                xlsx_filename = attachment[0]['filename']
                df = pd.read_excel(xlsx_data)
                df.drop(df[df['update'] >= pd.to_datetime('2023-05-01')].index, inplace=True)
                print(f'loaded attachment - {xlsx_filename}')
                columns_list = ['DLeft', 'start_month', 'start_day', 'start', 'ops station', 'o_road', 'update_month', 'update_day']
                models = md.create_models(df, columns_list)
                print('models created')
                print(f'models metrics - \n{models[1].to_string(index=True, line_width=80)}')
                print(f'models scores - \n{models[2].to_string(index=True, line_width=80)}')
                md.save_models(models)
                print('models saved')
                print('logs saved, no error found')
                logging.info('no errors found, models saved')
            except Exception as e:
                print('error found, see logs')
                logging.exception('error found: %', e)
            finally:
                send_logs(ms.master_mail)
        elif letter['sender'] == ms.master_mail and letter['subject'] == ms.prediction_subject:
            print('predicting data')
            logging.info('predicting data')
            try:
                attachment = load_xlsx_from_message(letter['message'])
                print('attachement found')
                xlsx_data = attachment[0]['data']
                xlsx_filename = attachment[0]['filename']
                df = pd.read_excel(xlsx_data)
                df.drop(df[df['update'] < pd.to_datetime('2023-05-01')].index, inplace=True)
                columns_list = ['DLeft', 'start_month', 'start_day', 'start', 'ops station', 'o_road', 'update_month', 'update_day']
                logging.info('Loading models')
                models = md.load_models()[0]
                logging.info('Models loaded')
                forecast = md.prediction(df, models, columns_list)
                print('forecast completed')
                update_trains = df_to_excel(forecast)
                print('updated data on trains compiled')
                logging.info('updated data on trains compiled')
                send_xlsx(ms.master_mail, 'updated data on trains', update_trains, 'update_trains.xlsx')
                print('updates sent, no error found')
                logging.info('updates sent, no error found')
                forecast.to_excel('update_trains.xlsx')
                logging.info('update saved locally')
            except Exception as e:
                print('error found, see logs')
                logging.exception('error found: %', e)
            finally:
                send_logs(ms.master_mail)


if __name__ == "__main__":
    main()

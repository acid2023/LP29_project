import email
import imaplib
from email.header import decode_header
import re
import mail_settings as ms
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
import logging
import modeling_settings as mds
from typing import Dict, List


def imap_login() -> imaplib.IMAP4_SSL:
    username = ms.username
    mail_pass = ms.mail_pass
    imap_server = ms.imap_server
    imap = imaplib.IMAP4_SSL(imap_server)
    imap.login(username, mail_pass)
    imap.select("INBOX")
    return imap


def stmp_login() -> smtplib.SMTP_SSL:
    username = ms.username
    mail_pass = ms.mail_pass
    smtp = smtplib.SMTP_SSL(ms.smtp_server)
    smtp.login(username, mail_pass)
    return smtp


def message_extract(imap: imaplib.IMAP4_SSL, request: bytes) -> email.message.Message:
    _, response = imap.fetch(request, '(RFC822)')
    msg = email.message_from_bytes(response[0][1])
    return msg


def get_message_by_id(message_id: bytes) -> email.message.Message:
    return message_extract(imap_login(), message_id)


def decode(bytes: bytes) -> str:
    try:
        return decode_header(bytes)[0][0].decode()
    except AttributeError:
        return bytes


def get_messages(all_messages=False) -> List:
    logging.info('getting messages from INBOX')
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
        subject = decode(msg['Subject'])
        logging.info(f'Message {i}, subject - {subject}')
        sender = re.search(email_pattern, msg['From']).group(1)
        msg_date = msg['Date']
        return_path = msg['Return-path']
        emails_list.append({'message_id': message_id,
                            'message': msg,
                            'subject': subject, 'date': msg_date,
                            'sender': sender, 'return_path': return_path})
        i += 1
    return emails_list


def archiveing_and_removing_messages(archive_list: List):
    logging.info("archieving and removing  messages")
    mail = imap_login()
    _, data = mail.search(None, 'ALL')
    message_nums = data[0].split()
    for message_num in message_nums:
        msg = message_extract(mail, message_num)
        if msg['Message-ID'] in archive_list:
            mail.copy(message_num, "ARCHIVE")
            logging.info(f"message from authorized user {msg['From']} copyied to ARCHIVE")
        else:
            mail.copy(message_num, "TRASH")
            logging.info(f"message from unauthorized user {msg['From']} copyied to TRASH")
        mail.store(message_num, "+FLAGS", '\\Deleted')
    logging.info("messages removed")


def load_xlsx_from_message(message: bytes) -> Dict:
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


def get_text_body_from_message(message) -> str:
    text_parts = []
    for part in message.walk():
        if part.get_content_type() == 'text/plain':
            text_parts.append(part.get_payload(decode=True).decode('utf-8'))
    text = '\n'.join(text_parts)
    return text


def get_columns_list_from_message(letter) -> List:
    default_columns = mds.DefaultColumns
    message = letter['message']
    letter_text = get_text_body_from_message(message)
    pattern = r'\[(.*?)\]'
    match = re.search(pattern, letter_text)
    if match:
        column_str = match.group(1)
        columns = [col.replace("’", "'").replace("‘", "'").strip("\"'") for col in column_str.split(',')]
        return columns
    else:
        return default_columns


def send_letter(receiver_address: str, subject: str, **kwargs) -> None:
    mail = stmp_login()
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = ms.username
    msg['To'] = receiver_address
    message_type = kwargs.get('message_type')
    attachment_data = kwargs.get('attachment')
    filename = kwargs.get('filename')
    message = kwargs.get('message')
    if message_type == 'xlsx':
        if not attachment_data:
            raise ValueError('data is empty')
        attachment = MIMEApplication(attachment_data, _subtype='xlsx')
        attachment['Content-Disposition'] = f'attachment; filename="{filename}"'
        msg.attach(attachment)
    elif message_type == 'logs':
        if not filename:
            raise ValueError('no filename for logs')
        with open(filename, 'rb') as file:
            session_log = file.read()
        attachment = MIMEApplication(session_log, _subtype='plain')
        attachment['Content-Disposition'] = f'attachment; filename="{filename}"'
        msg.attach(attachment)
    elif message_type == 'message':
        if not message:
            raise ValueError('message is empty')
        msg.attach(MIMEText(message, 'plain'))
    else:
        raise ValueError('invalid message_type')
    mail.sendmail(ms.username, receiver_address, msg.as_bytes())


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


def send_message(receiver_address, subject, message):
    mail = stmp_login()
    my_address = ms.username
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = my_address
    msg['To'] = receiver_address
    msg.attach(MIMEText(message, 'plain'))
    mail.sendmail(my_address, receiver_address, msg.as_string())


def mail_folders_setup():
    mail = imap_login()
    logging.info('setting up folders in INBOX')
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


if __name__ == "__main__":
    pass

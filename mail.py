import email
import imaplib
from email.header import decode_header
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText

import logging
import re

from typing import Dict, List, Union, IO
from datetime import datetime

import mail_settings as ms


def imap_login() -> imaplib.IMAP4_SSL:
    imap = imaplib.IMAP4_SSL(ms.imap_server)
    imap.login(ms.username, ms.mail_pass)
    imap.select("INBOX")
    return imap


def stmp_login() -> smtplib.SMTP_SSL:
    smtp = smtplib.SMTP_SSL(ms.smtp_server)
    smtp.login(ms.username, ms.mail_pass)
    return smtp


def message_extract(imap: imaplib.IMAP4_SSL, request: bytes) -> email.message.Message:
    _, response = imap.fetch(request, '(RFC822)')
    return email.message_from_bytes(response[0][1])


def get_message_by_id(message_id: bytes) -> email.message.Message:
    return message_extract(imap_login(), message_id)


def decode(bytes: bytes) -> str:
    try:
        return decode_header(bytes)[0][0].decode()
    except AttributeError:
        return bytes


def get_sender(msg_from: str):
    email_pattern = r'<([^<>]+)>'
    return re.search(email_pattern, msg_from).group(1)


def get_messages(all_messages=False) -> List[Dict[str, Union[bytes, email.message.Message, str, datetime]]]:

    logging.info('getting messages from INBOX')

    mail = imap_login()
    mail.select('INBOX')

    emails_list = []
    search_querry = 'ALL' if all_messages else 'UNSEEN'
    _, data = mail.search(None, search_querry)
    message_nums = data[0].split()

    for num, message_num in enumerate(message_nums):
        msg = message_extract(mail, message_num)
        message_id = msg['Message-ID']
        subject = decode(msg['Subject'])
        logging.info(f'Message {num}, subject - {subject}')
        sender = get_sender(msg['From'])
        msg_date = msg['Date']
        return_path = msg['Return-path']
        emails_list.append({'message_id': message_id,
                            'message': msg,
                            'subject': subject,
                            'date': msg_date,
                            'sender': sender,
                            'return_path': return_path})
    return emails_list


def archiveing_and_removing_messages(archive_list: List[bytes]) -> None:
    logging.info("archieving and removing  messages")
    mail = imap_login()
    _, data = mail.search(None, 'ALL')
    message_nums = data[0].split()
    for message_num in message_nums:
        msg = message_extract(mail, message_num)
        if msg['Message-ID'] in archive_list:
            mail.copy(message_num, "ARCHIVE")
            logging.info(f"message from authorized user {get_sender(msg['From'])} copyied to ARCHIVE")
        else:
            mail.copy(message_num, "TRASH")
            logging.info(f"message from unauthorized user {get_sender(msg['From'])} copyied to TRASH")
        mail.store(message_num, "+FLAGS", '\\Deleted')
    logging.info("messages removed")


def get_data_from_message(message: bytes, **kwargs: str) -> Union[str, bytes] | List[Dict[str, Union[str, IO]]]:
    get_type = kwargs.get('get_type')
    if get_type == 'text':
        text_parts = []
        for part in message.walk():
            payload = part.get_payload(decode=False)
            if part.get_content_type() == 'text/plain':
                if isinstance(payload, bytes):
                    text = payload.decode('utf-8')
                else:
                    text = payload
                text_parts.append(text)
        text = '\n'.join(text_parts)
        return text
    elif get_type == 'xlsx':
        xlsx_files = []
        for part in message.walk():
            try:
                filename = decode_header(part.get_filename())[0][0].decode() if part.get_filename() else ""
            except AttributeError:
                filename = part.get_filename() if part.get_filename() else ""
            if filename.endswith('.xlsx'):
                xlsx_data = part.get_payload(decode=True)
                xlsx_files.append({'filename': filename, 'data': xlsx_data})
        return xlsx_files
    elif get_type == 'signature':
        text_parts = []
        for part in message.walk():
            if part.get_content_type() == 'text/plain':
                text_parts.append(part.get_payload(decode=True).decode('utf-8'))
            text = '\n'.join(text_parts)
        return text
    else:
        raise ValueError('Invalid get_type argument')


def send_letter(receiver_address: str, subject: str, **kwargs: Union[str, IO]) -> None:
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


def mail_folders_setup() -> None:
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

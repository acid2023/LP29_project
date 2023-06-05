import email
import imaplib
from email.header import decode_header
import base64
import re
import mail_settings as ms
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
import io


def imap_login():
    username = ms.username
    mail_pass = ms.mail_pass
    imap_server = "imap.mail.ru"
    imap = imaplib.IMAP4_SSL(imap_server)
    imap.login(username, mail_pass)
    imap.select("INBOX")
    return imap


def stmp_login():
    server = smtplib.SMTP_SSL('smtp.mail.ru', 465)
    username = ms.username
    mail_pass = ms.mail_pass
    server = 'smtp.mail.ru'
    smtp = smtplib.SMTP_SSL(server)
    smtp.login(username, mail_pass)
    return smtp


def get_message_by_id(message_id):
    imap = imap_login()
    search_querry = f'HEADER Message-ID "{message_id}"'
    _, data = imap.search(None, search_querry)
    message_id = data[0].split()[0]
    _, data = imap.fetch(message_id, '(RFC822)')
    raw_email = data[0][1]
    return email.message_from_bytes(raw_email)


def get_messages(all_messages=False):
    email_pattern = r'<([^<>]+)>'
    mail = imap_login()
    mail.select('INBOX')
    emails_list = []
    search_querry = 'ALL' if all_messages else 'UNSEEN'
    _, data = mail.search(None, search_querry)
    message_nums = data[0].split()
    for message_num in message_nums:
        _, data = mail.fetch(message_num, '(RFC822)')
        raw_email = data[0][1]
        msg = email.message_from_bytes(raw_email)
        message_id = msg['Message-ID']
        subject = msg['Subject']
        subject = decode_header(subject)[0][0].decode()
        sender = msg['From']
        sender = re.search(email_pattern, sender).group(1)
        msg_date = msg['Date']
        return_path = msg['Return-path']
        emails_list.append({'message_id': message_id,
                            'subject': subject, 'date': msg_date,
                            'sender': sender, 'return_path': return_path})
    return emails_list


def load_xlsx_from_message(message):
    xlsx_files = []
    for id_part, part in enumerate(message.walk()):
        # content_type = part.get_content_type()
        filename = part.get_filename()
        if filename:
            filename = decode_header(filename)[0][0].decode()
            if filename.find('.xlsx') != -1:
                xlsx_data = part.get_payload(decode=True)
                xlsx_files.append({'filename': filename, 'data': xlsx_data})
    return xlsx_files


def send_attachment(receiver_address, subject, data, filename):
    mail = stmp_login()
    my_address = ms.username
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = my_address
    msg['To'] = receiver_address
    part = MIMEBase('application', "octet-stream")
    encoded_data = base64.b64encode(data)
    part.set_payload(encoded_data)
    part['Content-Disposition'] = f'attachment; filename="{filename}"'
    msg.attach(part)
    mail.sendmail(my_address, receiver_address, msg.as_string())


def DF_to_excel(df):
    excel_data = io.BytesIO()
    df.to_excel(excel_data, index=False)
    excel_data.seek(0)
    return excel_data.read()


def process_DF(df):
    df.dropna(inplace=True)
    return df
    # this will be my machine learning module - prediction based on last
    # fitting, updating models, etc


def main():
    emails = get_messages(all_messages=True)
    print(f'recieved {len(emails)} letters')
    for letter in emails:
        if letter['sender'] == ms.master_mail:
            message_id = letter['message_id']
            return_path = letter['return_path']
            msg = get_message_by_id(message_id)
    # here I know for sure that only one xlsx is attached
    attachment = load_xlsx_from_message(msg)
    xlsx_data = attachment[0]['data']
    xlsx_filename = attachment[0]['filename']
    print(f'loaded attachment - {xlsx_filename}')
    df_from_mail = pd.read_excel(xlsx_data)
    print(len(df_from_mail))
    return_df = process_DF(df_from_mail)
    print(len(return_df))
    print('sending back')
    return_data = DF_to_excel(return_df)
    send_attachment(return_path, 'return data', return_data, 'return.xlsx')
    print('completed')


if __name__ == "__main__":
    main()

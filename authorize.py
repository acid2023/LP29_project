import hashlib
import hmac
import base64
import mail_settings as ms
import re
import logging
from mail import get_data_from_message
import email


'''
def get_text_body_from_message(message: email.message.Message) -> str:
    text_parts = []
    for part in message.walk():
        if part.get_content_type() == 'text/plain':
            text_parts.append(part.get_payload(decode=True).decode('utf-8'))
    text = '\n'.join(text_parts)
    return text
'''


def generate_signature(username: str) -> bytes:
    username_bytes = username.encode('utf-8')
    master_key = ms.master_key.encode('utf-8')
    sha256_hash = hashlib.sha256(username_bytes)
    hmac_hash = hmac.new(master_key, sha256_hash.digest(), 'MD5')
    signature = base64.b64encode(hmac_hash.digest())
    return signature


def get_signature(letter: object) -> bytes:
    pattern = r'signature=[\'"]b\'([\w+/=]+)\'[\'"]'
    message = letter['message']
    letter_text = get_data_from_message(message, get_type='text')
    match = re.search(pattern, letter_text)
    if match:
        signature = match.group(1).strip()
        return signature.encode('utf-8')
    else:
        return None


def verify_signature(username: str, signature: bytes) -> bool:
    expected_signature = generate_signature(username)
    if expected_signature is not None:
        return hmac.compare_digest(signature, expected_signature)
    else:
        return False


def authorize_user(letter: email.message.Message) -> bool:
    signature = get_signature(letter)
    if signature is not None:
        sender = letter['sender']
        return_path = letter['return_path']
        if verify_signature(sender, signature) or verify_signature(return_path, signature):
            return True
        elif sender == ms.master_mail or return_path == ms.master_mail:
            return True
        else:
            return False
    return False


def generate_new_user_signature(letter: email.message.Message) -> bytes | None:
    pattern = r'new_user\s*=\s*[«"](.*?)[»"]'
    message = letter['message']
    letter_text = get_data_from_message(message, get_type='text')
    match = re.search(pattern, letter_text)
    if match:
        email_address = match.group(1)
        signature = generate_signature(email_address)
        logging.info('sugnature generated')
        return signature
    else:
        logging.info('signature was not generated')
        return None


if __name__ == "__main__":
    pass

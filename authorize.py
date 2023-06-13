import hashlib
import hmac
import base64
import mail_settings as ms
import re
import logging


def get_text_body_from_message(message):
    text_parts = []
    for part in message.walk():
        if part.get_content_type() == 'text/plain':
            text_parts.append(part.get_payload(decode=True).decode('utf-8'))
    text = '\n'.join(text_parts)
    return text


def generate_signature(username):
    username_bytes = username.encode('utf-8')
    master_key = ms.master_key.encode('utf-8')
    sha256_hash = hashlib.sha256(username_bytes)
    hmac_hash = hmac.new(master_key, sha256_hash.digest(), 'MD5')
    signature = base64.b64encode(hmac_hash.digest())
    return signature


def get_signature(letter):
    pattern = r'signature=[\'"]b\'([\w+/=]+)\'[\'"]'
    message = letter['message']
    letter_text = get_text_body_from_message(message)
    match = re.search(pattern, letter_text)
    if match:
        signature = match.group(1).strip()
        return signature.encode('utf-8')
    else:
        return None


def verify_signature(username, signature):
    expected_signature = generate_signature(username)
    if expected_signature is not None:
        return hmac.compare_digest(signature, expected_signature)
    else:
        return False


def authorize_user(letter):
    signature = get_signature(letter)
    if signature is not None:
        sender = letter['sender']
        return_path = letter['return_path']
        if verify_signature(sender, signature) or verify_signature(return_path, signature):
            logging.info('User %s authorized' % sender)
            print('User %s authorized' % sender)
            return True
        else:
            logging.info('User %s not authorized' % sender)
            print('User %s is not authorized' % sender)
            return False
        
    return False


def generate_new_user_signature(letter):
    pattern = r'new_user\s*=\s*[«"](.*?)[»"]'
    message = letter['message']
    letter_text = get_text_body_from_message(message)
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

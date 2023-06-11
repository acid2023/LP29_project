import hashlib
import hmac
import base64
import mail_settings as ms


def generate_signature(username):
    username_bytes = username.encode('utf-8')
    sha256_hash = hashlib.sha256(username_bytes)
    hmac_hash = hmac.new(ms.master_key, sha256_hash.digest(), 'MD5')
    signature = base64.b64encode(hmac_hash.digest())
    return signature


def verify_signature(username, signature):
    expected_signature = generate_signature(username)
    return hmac.compare_digest(signature, expected_signature)


if __name__ == "__main__":
    pass

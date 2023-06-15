import logging
import telegram
from telegram.error import TelegramError
import mail_settings as ms


class TelegramBotHandler(logging.Handler):
    def __init__(self, bot: telegram.Bot):
        super().__init__()
        self.bot = bot
        self.chat_id = ms.my_chat

    def emit(self, record):
        try:
            message = self.format(record)
            self.bot.send_message(chat_id=self.chat_id, text=message)
        except TelegramError as e:
            logging.info('Error sending log message to Telegram: %s' % e)


if __name__ == "__main__":
    pass

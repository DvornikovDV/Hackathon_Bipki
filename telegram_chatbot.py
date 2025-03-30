# telegram_chatbot.py
import os
import telebot
import ctypes.util

def find_library_patch(name):
    if name == "c":
        return "msvcrt.dll"
    return None

if ctypes.util.find_library("c") is None:
    ctypes.util.find_library = lambda name: find_library_patch(name)

import whisper  # Теперь импорт происходит после патча
from abstracts import AbstractChatBot
from llm_module import RealLLM
import config
from pydub import AudioSegment

class TelegramChatBot(AbstractChatBot):
    def clear_history(self, message):
        """Очищает историю сообщений пользователя по команде /clear."""
        user_id = message.from_user.id
        self.conversation_history[user_id] = []  # Очищаем историю
        self.bot.send_message(user_id, "История сообщений очищена.")

    def __init__(self):
        self.bot = telebot.TeleBot(config.TELEGRAM_API_TOKEN)
        self.llm = RealLLM()
        self.whisper_model = whisper.load_model("base")
        self.conversation_history = {}

        # Регистрируем обработчики
        self.bot.message_handler(commands=["clear"])(self.clear_history)
        self.bot.message_handler(content_types=["text", "voice"])(self.handle_message)

    def handle_message(self, message):
        user_id = message.from_user.id

        # Инициализируем историю, если её ещё нет
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []

        # Обработка голосового сообщения
        if message.content_type == "voice":
            file_info = self.bot.get_file(message.voice.file_id)
            downloaded_file = self.bot.download_file(file_info.file_path)
            ogg_path = f"voice_{user_id}.ogg"
            wav_path = f"voice_{user_id}.wav"
            with open(ogg_path, "wb") as new_file:
                new_file.write(downloaded_file)
            # Используем Whisper для транскрипции
            transcript = self.handle_voice(user_id, ogg_path, wav_path)
            # Добавляем в историю сообщение от пользователя (тип voice – уже в виде текста)
            self.conversation_history[user_id].append(("User", transcript))
            # Формируем контекст из 3 предыдущих сообщений, если есть
            context = self.build_context(user_id, transcript)
            response = self.llm.process(context)
            # Добавляем ответ бота в историю
            self.conversation_history[user_id].append(("Bot", response))
            self.bot.send_message(user_id, f"Транскрипция: {transcript}\nОтвет: {response}")
            try:
                os.remove(ogg_path)
                os.remove(wav_path)
            except Exception as e:
                print("Ошибка удаления файлов:", e)
            return

        # Обработка текстового сообщения
        if message.content_type == "text":
            text = message.text.strip()
            # Обработка команд /start и /help отдельно
            lower_text = text.lower()
            if lower_text in ["/start", "/help"]:
                help_text = (
                    "Привет! Я чат-бот поликлиники.\n\n"
                    "Вы можете задать любой вопрос о расписании работы и ценах на услуги\n\n"
                    "Просто введите сообщение, и я постараюсь вам помочь."
                )
                self.bot.send_message(user_id, help_text)
                return

            # Добавляем текстовое сообщение пользователя в историю
            self.conversation_history[user_id].append(("User", text))
            # Формируем контекст для LLM: берем 3 последних сообщения из истории перед текущим
            context = self.build_context(user_id, text)
            response = self.llm.process(context)
            # Добавляем ответ бота в историю
            self.conversation_history[user_id].append(("Bot", response))
            self.bot.send_message(user_id, response)

    def handle_voice(self, user_id: int, ogg_path: str, wav_path: str) -> str:
        """
        Использует Whisper для транскрипции голосового сообщения:
         - Конвертирует ogg в wav.
         - Вызывает модель Whisper для получения транскрипта.
        """
        try:
            # Конвертируем ogg в wav
            print("Начало обработки аудио")
            sound = AudioSegment.from_file(ogg_path, format="ogg")
            sound.export(wav_path, format="wav")
            # Транскрибируем с помощью Whisper
            result = self.whisper_model.transcribe(wav_path, language="ru")
            transcript = result["text"].strip()
        except Exception as e:
            transcript = "Не удалось распознать голосовое сообщение."
            print("Ошибка распознавания:", e)
        return transcript

    def build_context(self, user_id: int, current_message: str) -> str:
        """
        Формирует контекст для LLM, включающий 3 последних сообщения (с указанием отправителя).
        Если истории меньше 3 сообщений, берутся все доступные.
        К текущему сообщению добавляется префикс "Пользователь:".
        """
        history = self.conversation_history.get(user_id, [])
        # Берем до 3 последних сообщений, исключая текущее (оно уже добавлено)
        context_messages = history[-3:]
        # Маппинг оригинальных меток на русские
        sender_labels = {"User": "Пользователь", "Bot": "Ответ"}
        # Формируем строки с русскими подписями
        context_lines = [
            f"{sender_labels.get(sender, sender)}: {text}"
            for sender, text in context_messages
        ]
        # Объединяем строки в один текст с переводами строк
        return "\n".join(context_lines)

    def start(self):
        self.bot.polling()

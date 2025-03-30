# abstracts.py
from abc import ABC, abstractmethod

class AbstractChatBot(ABC):
    @abstractmethod
    def handle_message(self, user_id: int, message: str) -> str:
        """
        Обрабатывает входящее сообщение от пользователя.
        """
        pass

class PolyclinicInterface(ABC):
    @abstractmethod
    def book_appointment(self, user_id: int, doctor_id: str, time_slot: str) -> str:
        """
        Оформляет запись к врачу для пользователя.
        """
        pass

class BaseLLM(ABC):
    @abstractmethod
    def process(self, input_text: str) -> str:
        """
        Обрабатывает входной текст и возвращает ответ.
        """
        pass

class RAGInterface(ABC):
    @abstractmethod
    def query(self, input_text: str) -> str:
        """
        Выполняет поиск и генерацию ответа с использованием RAG.
        """
        pass

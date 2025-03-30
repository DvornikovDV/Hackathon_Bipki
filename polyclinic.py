# polyclinic.py
from abstracts import PolyclinicInterface

class DummyPolyclinic(PolyclinicInterface):
    def book_appointment(self, user_id: int, doctor_id: str, time_slot: str) -> str:
        # Заглушка для записи: возвращает сообщение о том, что функция пока не реализована
        return (f"Запись к врачу (ID: {doctor_id}) на {time_slot} для пользователя {user_id} "
                "пока не реализована.")

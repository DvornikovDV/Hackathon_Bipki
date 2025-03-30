from abstracts import BaseLLM
import config
import requests


class RealLLM(BaseLLM):
    def process(self, input_text: str) -> str:
        # Если в запросе есть "health", выполняем GET-запрос
        if "health" in input_text.lower():
            return self.check_health()

        url = "http://127.0.0.1:8888/qa"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        data = {"question": input_text}

        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result.get("answer", "Извините, не удалось получить ответ.")
        except requests.exceptions.RequestException as e:
            return f"Ошибка при обращении к LLM! {e}"

    def check_health(self) -> str:
        url = "http://127.0.0.1:8888/health"
        headers = {"accept": "application/json"}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return "Система работает исправно!"
        except requests.exceptions.RequestException as e:
            return f"Ошибка проверки статуса LLM! {e}"

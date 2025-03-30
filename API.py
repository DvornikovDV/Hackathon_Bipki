import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from main_langchain import run_query

# Инициализация приложения
app = FastAPI()




# пример модели: описывает, что клиент должен отправить.
class UserRequest(BaseModel):
    question: str

# описывает, что сервер вернёт
class UserResponse(BaseModel):
    answer: str
    context: list[str]


# специальная функция, котора будет вызываться при старте сервера
@app.on_event("startup")
async def generate_openapi_yaml():
    # автоматическая генерации спецификации для API в yaml
    openapi_schema = app.openapi()
    with open("openapi.yaml", "w") as f:
        yaml.dump(openapi_schema, f, default_flow_style=False, allow_unicode=True)
    print("OpenAPI YAML сгенерирован при запуске сервера!")
    

# =====================
#  Эндпоинты
# =====================

# get-запрос для корневого эндпоинта, используется в качестве health-check
@app.get("/health")
async def health():
    """health-check; в норме выдаёт Ok"""
    # docstring будет виден в /doc
    return {"status": "Ok", "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.post("/qa", response_model=UserResponse,  summary="Запрос пользователя")
async def answer_to_user(user: UserRequest) -> UserResponse:
    """
    * .split('Полезный ответ: ')[1].split("Вопрос пользователя:")[0]
    """
    answer = run_query(user.question)
    return UserResponse(answer=answer["answer"], context=answer["context"])


# =====================
#  Запуск сервера
# =====================
# uvicorn API:app --host 0.0.0.0 --port 8888 
# --workers 4

# Документация будет доступна по следующим адресам:
# Swagger UI:
# 📌 http://127.0.0.1:8888/docs — Интерактивная документация.
# Там можно сразу протестировать запросы, отправить данные и посмотреть ответы.

# ReDoc:
# 📌 http://127.0.0.1:8888/redoc — Альтернативная, более минималистичная документация.

# Файл OpenAPI:
# 📌 http://127.0.0.1:8888/openapi.json — Спецификация в формате JSON.

"""
10.1.19.2
192.168.49.61

=====================
Модели запроса и ответа

Модель — это Python-класс, который наследуется от pydantic.BaseModel и описывает, какие поля должны быть в данных, какие у них типы, и какие правила валидации применяются.

FastAPI автоматически:
* Проверит, что данные соответствуют этим типам.
* Сгенерирует документацию для Swagger UI.
* Преобразует входные данные в объект Python

Модели отделяют логику работы с данными от логики обработки запросов. Код становится аккуратным и поддерживаемым.
=====================
"""
# 1. Импорты
# venv\Scripts\activate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate

from СкрапИОбработ import *

def read_file_to_list(file_path):
    """
    Читает файл построчно и возвращает список строк.

    :param file_path: Путь к файлу.
    :return: Список строк из файла.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Читаем файл построчно и удаляем символы переноса строки (\n)
            lines = [line.strip() for line in file]
        return lines
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден.")
        return []
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {e}")
        return []


def add_new_texts_to_db(texts, metadatas):
    """
    Функция для добавления новых записей

    :param texts: Список строк, новых данных.
    :param metadata: Список словарей, содержащих метаданные.
    """

    #временно /------------------------------------------------------------------/
    for i in metadatas:
        i["tags"] = ", ".join(i["tags"]) if i["tags"] else "нет тегов"

    vectorstore.add_texts(texts = texts, metadatas = metadatas)
    print(f"Добавлено {len(texts)} новых записей.")


def run_query(query):
    """
    Функция запроса к цепочке

    :param query: Строка запроса.
    :return: Словарь, содержащий ответ и контекст (исходные документы).
    """
    response = qa_chain.invoke(query)
    #.split("Вопрос пользователя:")[1].split('Полезный ответ: ')[1]
    answer = response["result"]  # Ответ модели
    source_documents = response["source_documents"]  # Исходные документы

    # Извлечение текстов из исходных документов
    context = [doc.page_content for doc in source_documents]

    return {
        "answer": answer,
        "context": context
    }


# 2. Инициализация модели эмбеддингов
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 3. Инициализация ChromaDB
persist_directory = "./chroma_db"
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)




# 5. Инициализация языковой модели (TinyLlama)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Создание pipeline для генерации текста
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,  # Ограничение длины ответа
    device="cpu", ############################################################################## device="CUDA" типа того
    temperature=0.3,  # Установите значение температуры
    do_sample=True    # Включите сэмплирование   
)
llm = HuggingFacePipeline(pipeline=pipe)

# Новый шаблон промпта
prompt_template = """Ты помощник-консультант поликлиники. Ты всегда учтив и вежлив. Твоя задача консультировать пользователей, используя информацию исключительно из предоставленной базы данных.
Если ответа в базе нет, то скажи, что не можешь помочь с данным вопросом, и не пытайся придумать ответ самостоятельно. 

Если пользователь спрашивает об услуге, то ты должен выдать полную информацию: Упомянуть название, рассказать о цене, и если в базе имеется дополнительная информация, то вставить в ответ и её.

Примеры взаимодействия, показывающие сценарии того, как ты должен себя вести (не вставляй их в диалог, только используй такую модель поведения):
Пример 1:
```
Пользователь: У вас можно провериться у лора?
Ответ: Да! У нас имеется специалист врач-оториноларинголог. Стоимость профилактического приёма (осмотр, консультация) 700 рублей.
```
Пример 2:
```
Пользователь: Здравствуйте, у вас есть кардиолог?
Ответ: Здравствйте. Да, у нас присутствует кардиолог. Стоимость приёма 2300 рублей.
Пользователь: А для ребенка столько же?
Ответ: Приём у деского кардиолога будет стоить по разному: первичный также 2300, а повторный 1600.
```

Контекст из базы данных:
{context}\n


Фрагмент беседы с пользователем (может содержать голосовые сообщения, если не удалось их расшифровать, то не обращай внимания на конкретное сообщение. Ты всегда должен отвечать только на самый последний вопрос пользователя, а остальные сообщения использовать как контекст диалога. Ты должен вывести только 1 ответ!): 

{question}

Ответ:
"""



PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 6. Создание цепочки для поиска и ответов
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"score_threshold": 0.5, "k": 10}),#
    return_source_documents=True,
    verbose=False,
    chain_type_kwargs={"prompt": PROMPT}  # Передаем новый шаблон
)


# 7. Добавление новых записей - время работы 

'''
scr = ClinicScraper(base_working_hours_url="https://clinica.chitgma.ru/informatsiya-po-otdeleniyu-9")
table = scr.scrape_working_hours()
print(table)
w_hours = []
for i in table:
    w_hours.append(i.to_str())

metadatas = [{
    "tags" : "Рабочие часы, работы"
}, {
    "tags" : "Рабочие часы, работы"
}, {
    "tags" : "Рабочие часы, работы"
}]

add_new_texts_to_db(w_hours, metadatas)
'''

# 7. Добавление новых записей - услуги
'''


scr = ClinicScraper(base_working_hours_url="https://clinica.chitgma.ru/informatsiya-po-otdeleniyu-9")
table = scr.scrape_services()
proces = MedicalDataProcessor()
ready_entries = proces.process_raw_data(table)

add_new_texts_to_db(ready_entries.texts, ready_entries.metadata)
'''
# Удаление всей коллекции при дублировании
# vectorstore.reset_collection()


# 8. Выполнение запроса


"""
исключить услуги льготников(в конце) -- мед осмотры

убрать кол-во( до 0.5 )

query = "Кто разработал этого чач бота?"
response = qa_chain.invoke(query)
question = response["query"]
answer = response["result"]
#.split('Полезный Ответ: ')[1].split("Вопрос Пользователя:")[0]
print()
print(question)
print()
print(answer)
"""



"""  -------------

texts = ["Чат бота для хакахака разработали 3 величайших программиста И 1 величайший мыслитель.", 
"Величайший программист 1: Непомнящих Станислав",
"Величайший программист 2: Ошлаков Данил, через О",
"Величайший программист 3: Дворников Даниил",
"Величайший мыслитель: Мнацаканян Артур"]

vectorstore.add_texts(texts)

"""
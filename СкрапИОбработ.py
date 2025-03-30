import json
from io import BytesIO
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pdfplumber


@dataclass
class WorkTimeInfo:
    _days: str  # День недели -> время приема
    _hours: str

    @property
    def days(self) -> str:
        return self._days

    @property
    def hours(self) -> str:
        return self.hours

    def to_dict(self) -> dict:
        return {
            self._days: self._hours,
        }

    def to_str(self) -> str:
        return f"{self._days} : {self._hours}"


@dataclass
class ServiceInfo:
    _article: int
    _code: str
    _name: str
    _price: float

    @property
    def name(self) -> str:
        return self._name

    @property
    def article(self) -> int:
        return self._article

    @property
    def code(self) -> str:
        return self._code

    @property
    def price(self) -> float:
        return self._price

    def formatted_info(self) -> str:
        return (f"Услуга: {self._name}\n"
                f"Артикул: {self._article}\n"
                f"Код: {self._code}\n"
                f"Цена: {self._price:.2f} руб.")

    def to_dict(self) -> dict:
        return {
            'name': self._name,
            'article': self._article,
            'code': self._code,
            'price': self._price
        }

    def to_str(self) -> str:
        return f"{self._name} Цена: {self._price} рублей"
'''
    def to_str(self) -> str:
        return f"Артикул: {self._article} | Код услуги: {self._code} | Наименование услуги: {self._name} | Цена: {self._price} рублей"
'''

@dataclass
class ReadyEntries:
    texts: List[str]
    metadata: List[Dict]


class AbstractDataProcessor(ABC):
    @abstractmethod
    def process_raw_data(self, raw_data: List) -> ReadyEntries:
        """Основной метод обработки сырых данных"""
        pass

    @abstractmethod
    def _categorize_services(self, services: List) -> List:
        """Категоризация услуг (должен быть реализован в подклассах)"""
        pass


class AbstractClinicScraper(ABC):
    @abstractmethod
    def scrape_working_hours(self, url: str) -> List[WorkTimeInfo]:
        """Получить режим работы поликлиники"""
        pass

    @abstractmethod
    def scrape_services(self) -> List[ServiceInfo]:
        """Получить перечень услуг и их стоимость"""
        pass

    @abstractmethod
    def process_for_chroma(self, url: str) -> List[Dict]:
        """Обработать все данные для сохранения в ChromaDB"""
        pass


import requests
import re
from bs4 import BeautifulSoup, NavigableString
from typing import Optional, Dict, List


class ClinicScraper(AbstractClinicScraper):
    def __init__(self, base_working_hours_url: str = "https://clinica.chitgma.ru",
                 pdf_url: str = "https://clinica.chitgma.ru/images/Preyskurant/2025/1DP.pdf",
                 categories_config: str = 'medical_service_categories.json'):
        self.pdf_url = pdf_url
        self.categories_config = categories_config
        self.base_working_hours_url = base_working_hours_url

    def scrape_working_hours(self, url: str = None) -> Optional[List[WorkTimeInfo]]:
        """Реализация метода для извлечения информации о режиме работы"""
        if url is None:
            url = self.base_working_hours_url
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            return self._extract_working_hours(soup)
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе: {e}")
            return None

    def scrape_services(self) -> List[ServiceInfo]:
        """Извлечение информации об услугах и ценах"""
        try:
            return self._extract_services_from_pdf()
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе: {e}")
            return []

    def process_for_chroma(self, url: str) -> List[Dict]:
        pass

    # Вспомогательные методы
    def _extract_working_hours(self, soup) -> Optional[List[WorkTimeInfo]]:
        """Вспомогательный метод для извлечения информации о режиме работы"""
        text_block = soup.find(string=re.compile("Режим работы отделения:"))
        if not text_block:
            return None

        working_hours_text = ""
        for elem in text_block.parent.next_elements:
            if isinstance(elem, NavigableString):
                working_hours_text += str(elem)
            if elem.name == "p" and "Предварительная запись" in elem.get_text():
                break

        # Очищаем текст от HTML тегов и &nbsp;
        working_hours_text = re.sub('<.*?>', '', working_hours_text)
        working_hours_text = working_hours_text.replace('&nbsp;', ' ').strip()

        # Разбиваем текст на строки и обрабатываем
        lines = working_hours_text.split('\n')
        working_hours = []

        for line in lines:
            line = line.strip()
            if not line or "Режим работы отделения:" in line:
                continue

            # Разделяем строку на день недели и время
            parts = re.split(r'\s{2,}', line)  # Разделяем по двум и более пробелам
            if len(parts) >= 2:
                day = parts[0].strip()
                time = parts[-1].strip()
                working_hours.append(WorkTimeInfo(day, time))
            elif len(parts) == 1 and parts[0]:
                # Если строка не разделена (возможно, из-за другого форматирования)
                # Пробуем разделить по первому вхождению цифры
                match = re.search(r'(\D+)(\d.+)$', parts[0])
                if match:
                    day = match.group(1).strip()
                    time = match.group(2).strip()
                    working_hours.append(WorkTimeInfo(day, time))

        return working_hours

    def _extract_services_from_pdf(self) -> Optional[List[ServiceInfo]]:
        """Извлекает таблицу из PDF"""
        try:
            response = requests.get(self.pdf_url)
            response.raise_for_status()
            all_rows = []

            with pdfplumber.open(BytesIO(response.content)) as pdf:
                first_page = pdf.pages[0]
                first_table = first_page.extract_tables()

                if not first_table or len(first_table[0]) == 0:
                    return None

                headers = first_table[0][0]
                last_row = first_table[0][1]
                if not headers:
                    return None

                for page in pdf.pages:
                    tables = page.extract_tables()
                    if not tables:
                        continue

                    for table in tables:
                        start_idx = 1 if page == first_page else 0
                        for row in table[start_idx:]:
                            if len(row) == len(headers):
                                if last_row and (row[2] is None):
                                    new_name = last_row[2]
                                    row[2] = new_name.replace("первичный", "повторный")
                                service = self._process_raw_service(row)
                                if service is None:
                                    continue
                                all_rows.append(service)
                                last_row = row

            return all_rows if all_rows else None

        except Exception as e:
            print(f"Ошибка при извлечении таблицы: {e}")
            return None

    def _process_raw_service(self, raw_data: List) -> Optional[ServiceInfo]:
        """Обрабатывает сырые данные об услугах"""
        empty_fields = sum(1 for v in raw_data if v == '')
        if empty_fields > 2:
            return None

        price_str = raw_data[3].replace(' ', '') if raw_data[3] else "0"
        try:
            price = float(price_str)
        except (ValueError, AttributeError):
            price = 0.0
        processed = ServiceInfo(raw_data[0], raw_data[1], raw_data[2], price)

        return processed


class MedicalDataProcessor(AbstractDataProcessor):
    def __init__(self, categories_config: str = 'medical_service_categories.json'):
        self.categories_config = categories_config

    def process_raw_data(self, raw_data: List[ServiceInfo]) -> ReadyEntries:
        """Обработка сырых данных в готовый формат для сохранения"""
        # 1. Категоризация и преобразование в ServiceInfo
        services = self._categorize_services(raw_data)

        # 2. Формирование текстов и метаданных
        texts = []
        metadata = []

        for service in services:
            texts.append(service[0])
            metadata.append(service[1])

        return ReadyEntries(texts=texts, metadata=metadata)

    def _categorize_services(self, services: List[ServiceInfo]) -> [str, Dict]:
        """Категоризация медицинских услуг (возвращает все подходящие категории)"""
        try:
            with open(self.categories_config, 'r', encoding='utf-8') as f:
                categories = json.load(f)['medical_service_categories']
        except Exception as e:
            print(f"Ошибка загрузки конфига категорий: {e}")
            categories = {}

        result = []
        for service in services:
            service_name = service.name.lower()
            categories_list = []  # Здесь будут все подходящие категории

            # Проверяем все категории
            for cat, keywords in categories.items():
                if any(kw.lower() in service_name for kw in keywords.split(", ")):
                    categories_list.append(cat)  # Добавляем все подходящие категории

            # Если ни одна категория не подошла, ставим "другое"
            if not categories_list:
                categories_list = ["другое"]

            # Формируем словарь с категориями
            cats = {
                "tags": categories_list,  # Теперь это список, а не строка
                "category": "услуга"
            }

            result.append((service.to_str(), cats))

        return result

"""

scr = ClinicScraper(base_working_hours_url="https://clinica.chitgma.ru/informatsiya-po-otdeleniyu-9")
table = scr.scrape_services()
proces = MedicalDataProcessor()
ready_entries = proces.process_raw_data(table)
for text, metadata in zip(ready_entries.texts, ready_entries.metadata):
    print(f"Текст: '{text}', Мета: {metadata}")
    print(f"-------------------------------------------------------------------")



"""
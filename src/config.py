from pathlib import Path

# Настройка путей
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = DATA_DIR / "cr_manifest.json"

# Настройки API
API_URL = 'https://apicr.minzdrav.gov.ru/api.ashx'
PARAMS = {'op': 'GetJsonClinrecsFilterV2'}

HEADERS = {
    'accept': 'application/json, text/plain, */*',
    'content-type': 'application/json',
    'origin': 'https://cr.minzdrav.gov.ru',
    'referer': 'https://cr.minzdrav.gov.ru/',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36',
}

# Payload для фильтрации по нозологиям
PAYLOAD = {
    'filters': [
        {
            'fieldName': 'status',
            'filterType': 1,
            'filterValueType': 2,
            'value1': 0,
            'value2': '',
            'values': [],
        },
        {
            'fieldName': 'mkbid',
            'filterType': 9,
            'filterValueType': 1,
            'value1': '',
            'value2': '',
            'values': [3712, 2000],
        },
    ],
    'sortOption': {'fieldName': 'publishdate', 'sortType': 2},
    'pageSize': 120,
    'currentPage': 1,
    'useANDoperator': True,
    'columns': [],
}

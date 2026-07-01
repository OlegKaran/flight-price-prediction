import pandas as pd
import requests
import time
from itertools import permutations
from api_aviasales import API_TOKEN

url = 'http://api.travelpayouts.com/v2/prices/month-matrix'
headers = {
    'x-access-token': API_TOKEN,
    'User_Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Version/17.0 Safari/537.36'
}

cities = ['MOW', 'LED', 'BCN', 'HKT', 'IST', 'DXB', 'DOH', 'AYT', 'BKK', 'LON', 'PAR', 'TBS', 'BEG', 'PEK', 'NRT', 'FRA', 'AMS', 'AER']
routes = list(permutations(cities, 2))
print(f'Всего маршрутов для проверки: {len(routes)}')

months_to_collect = [
    '2026-06-01', '2026-07-01', '2026-08-01', '2026-09-01',
    '2026-10-01', '2026-11-01', '2026-12-01', '2027-01-01',
    '2027-02-01', '2027-03-01', '2027-04-01', '2027-05-01',
]

start_time = time.time()
all_tickets = []
for origin, destination in routes:
    for target_month in months_to_collect:
        params = {
            'origin': origin,
            'destination': destination,
            'month': target_month,
            'show_to_affiliates': 'true'
        }
        max_retries = 3
        for attempt in range (max_retries):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json().get('data', [])
                    all_tickets.extend(data)
                    print(f'Успешно: {origin} -> {destination} (Найдено билетов: {len(data)})')
                    break
                elif response.status_code == 429:
                    print('Превышен лимит запросов в минуту (Ошибка 429)! Ждем 60 секунд...')
                    time.sleep(60)
                else:
                    print(f'Ошибка {response.status_code} для {origin} -> {destination}')
                    break
            except requests.exceptions.ConnectionError as e:
                print(f'Потеряно соединение ({origin} -> {destination}). Попытка {attempt+1} из {max_retries}...')
                time.sleep(5)
            except requests.exceptions.Timeout:
                print(f'Сервер слишком долго не отвечает {origin} -> {destination}. Попытка {attempt+1}')
                time.sleep(5)
        time.sleep(0.25)
main_data = pd.DataFrame(all_tickets)
end_time = time.time()
total_seconds = end_time - start_time
print(f'Общее время работы: {total_seconds // 60} минут {total_seconds % 60} секунд')
print(f'Сбор данных завершен! Всего строк в датасете: {len(main_data)}')
main_data.to_pickle('aviasales_dataset_in_rubles_with_months(1).pkl')
print('Данные успешно сохранены в aviasales_dataset_in_rubles_with_months(1).pkl')
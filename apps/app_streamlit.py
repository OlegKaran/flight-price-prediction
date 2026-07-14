import os
import datetime
import requests
import streamlit as st
from config import CITY_MAPPING

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title('Предсказание цен на авиабилеты')
st.write('Выберите параметры перелета, и XGBoost рассчитает примерную стоимость авиабилетов')

today = datetime.date.today()
max_allowed_date = today + datetime.timedelta(days=365)

col1, col2 = st.columns(2)
with col1:
    depart_date = st.date_input(
        'Дата вылета',
        value=today,
        min_value=today,
        max_value=max_allowed_date
    )
    origin = st.selectbox(
        'Город вылета',
        options=list(CITY_MAPPING.keys()),
        format_func=lambda x: CITY_MAPPING[x]
    )
with col2:
    destination = st.selectbox(
        'Город прибытия',
        options=list(CITY_MAPPING.keys()),
        format_func=lambda x: CITY_MAPPING[x]
    )
    number_of_changes = st.slider(
        label='Количество пересадок', min_value=0, max_value=5, value=0, step=1
    )

if st.button('Узнать цену', type='primary'):
    if origin == destination:
        st.error('Ошибка: город вылета и город прилета не могут совпадать! Выберите разные города!')
    else:
        payload = {
            'depart_date': depart_date.isoformat(),
            'origin': CITY_MAPPING[origin],
            'destination': CITY_MAPPING[destination],
            'number_of_changes': number_of_changes,
        }
        try:
            with st.spinner('Считаем цену...'):
                response = requests.post(f'{API_URL}/predict', json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Ожидаемая цена билета: {result['predicted_price']} ₽")
                st.caption(
                    f"Маршрут: {result['route']} · "
                    f"время предсказания: {result['prediction_time_ms']} мс"
                )
            elif response.status_code == 404:
                st.error(response.json().get('detail', 'Маршрут не найден'))
            elif response.status_code == 422:
                st.error('Некорректные данные запроса. Проверьте выбранные параметры.')
            else:
                st.error(f'Сервис вернул ошибку {response.status_code}. Попробуйте позже.')

        except requests.exceptions.ConnectionError:
            st.error(f'Не удалось подключиться к API ({API_URL}). Проверьте, что сервис запущен.')
        except requests.exceptions.Timeout:
            st.error('Сервис слишком долго не отвечает. Попробуйте ещё раз.')

import streamlit as st
import pandas as pd
import datetime
from ml_service import FlightPricePredictor
from config import FLIGHT_INFO, CITY_MAPPING

flight_price_predict = FlightPricePredictor("models_and_pipelines/flight_price_predictor.cbm",
                                               "models_and_pipelines/preprocessor.pkl",
                                            )
@st.cache_resource
def load_model_and_artifacts():
    return flight_price_predict.model, flight_price_predict.preprocessor

model, preprocessor = load_model_and_artifacts()
st.title('️Предсказание цен на авиабилеты')
st.write('Выберите параметры перелета, и CatBoost рассчитает примерную стоимость авиабилетов')
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
    origin = st.selectbox('Город вылета',
                          options=list(CITY_MAPPING.keys()),
                          format_func=lambda x: CITY_MAPPING[x]
    )
with col2:
    destination = st.selectbox('Город пребывания',
                               options=list(CITY_MAPPING.keys()),
                               format_func=lambda x: CITY_MAPPING[x]
    )
    number_of_changes = st.slider(label='Количество пересадок', min_value=0, max_value=5, value=0, step=1)

if st.button('Узнать цену', type='primary'):
    if origin == destination:
        st.error('Ошибка: город вылета и город прилета не могут совпадать! Выберите разные города!')
    else:
        current_route_key = f'{origin}-{destination}'
        route_features = FLIGHT_INFO[current_route_key]
        distance = route_features['distance']
        duration = route_features['duration']
        input_data = pd.DataFrame({
            'depart_date': [depart_date],
            'origin': [origin],
            'destination': [destination],
            'found_at': [today],
            'number_of_changes': [number_of_changes],
            'duration': [duration],
            'distance': [distance]
        })
        final_data = flight_price_predict.prepare_data(input_data)
        prediction = flight_price_predict.model_prediction(final_data)
        st.success(f'Ожидаемая цена билета: {prediction} ₽')


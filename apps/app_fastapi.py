from fastapi import FastAPI, HTTPException
import pandas as pd
from datetime import date
import time
from schemas import FlightInfo, PriceResponse
from ml_service import FlightPricePredictor
from config import CITY_TO_IATA, FLIGHT_INFO

flight_price_prediction = FlightPricePredictor("models_and_pipelines/flight_price_predictor_xgboost.json",
                                               "models_and_pipelines/preprocessor.pkl",
                                               )

app = FastAPI(title="Flight_price_predictor")

@app.get("/")
def read():
    return {"message": "Сервер работает"}

@app.post("/predict", response_model=PriceResponse)
def predict_price(request: FlightInfo):
    iata_code_of_origin = CITY_TO_IATA.get(request.origin)
    iata_code_of_destination = CITY_TO_IATA.get(request.destination)
    route_key = f"{iata_code_of_origin}-{iata_code_of_destination}"
    route_data = FLIGHT_INFO.get(route_key)
    if not route_data:
        raise HTTPException(
            status_code=404,
            detail=f'Маршрут {route_key} не найден в базе данных. Выберите другой маршрут'
        )
    actual_distance = route_data["distance"]
    actual_duration = route_data["duration"]
    input_data = pd.DataFrame({
        "depart_date": [request.depart_date],
        "origin": [iata_code_of_origin],
        "destination": [iata_code_of_destination],
        "found_at": [date.today()],
        "number_of_changes": [request.number_of_changes],
        "duration": [actual_duration],
        "distance": [actual_distance]
    })
    processed_data = flight_price_prediction.prepare_data(input_data)
    start_time = time.perf_counter()
    prediction = flight_price_prediction.model_prediction(processed_data)
    end_time = time.perf_counter()
    time_to_predict = end_time - start_time
    return {
        "found_at": f"{date.today()}",
        "depart_date": f"{request.depart_date}",
        "route": f"{request.origin} ({iata_code_of_origin}) -> {request.destination} ({iata_code_of_destination})",
        "number_of_changes": request.number_of_changes,
        "predicted_price": prediction,
        "prediction_time_ms": f"{(time_to_predict*1000):.3f}"
    }



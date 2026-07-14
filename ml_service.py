import joblib
import pandas as pd
from catboost import CatBoostRegressor


class FlightPricePredictor:
    def __init__(self, model_path, preprocessor_path):
        self.model = CatBoostRegressor()
        self.model.load_model(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def prepare_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        processed_df = input_data.copy()
        processed_df['depart_date'] = pd.to_datetime(processed_df['depart_date'], utc=True)
        processed_df['found_at'] = pd.to_datetime(processed_df['found_at'], utc=True)
        processed_df['departure_day_of_week'] = processed_df['depart_date'].dt.dayofweek
        processed_df['departure_day'] = processed_df['depart_date'].dt.day
        processed_df['departure_month'] = processed_df['depart_date'].dt.month
        processed_df['departure_year'] = processed_df['depart_date'].dt.year
        processed_df['time_diff'] = processed_df['depart_date'] - processed_df['found_at']
        processed_df['days_to_departure'] = processed_df['time_diff'].dt.days
        processed_df.drop(columns=['depart_date', 'found_at', 'time_diff'], inplace=True)
        bins_season = [0, 2, 5, 8, 11, 12]
        labels_season = ["Winter", "Spring", "Summer", "Autumn", "Winter"]
        processed_df['season'] = pd.cut(processed_df['departure_month'], bins=bins_season, labels=labels_season,
                                        ordered=False)
        processed_df['is_weekend'] = (processed_df['departure_day_of_week'] >= 5).astype(int)
        processed_df['departure_year'] = processed_df['departure_year'] - 2026
        if 'departure_day' in processed_df.columns:
            bins = [0, 10, 20, 32]
            month_parts = ['start', 'middle', 'end']
            processed_df['month_part'] = pd.cut(processed_df['departure_day'], bins=bins, labels=month_parts)
            processed_df.drop(columns='departure_day', inplace=True)
        # cols_to_normalize = ['duration', 'distance', 'days_to_departure']
        # cols_to_one_hot_encode = ['month_part', 'departure_month', 'departure_day_of_week', 'origin', 'destination', 'season']
        # processed_df[cols_to_normalize] = processed_df[cols_to_normalize].astype(float)
        # processed_df.loc[:, cols_to_normalize] = self.scaler.transform(processed_df[cols_to_normalize])
        # encoded_columns = self.encoder.transform(processed_df[cols_to_one_hot_encode])
        # encoded_df = pd.DataFrame(
        #     encoded_columns,
        #     columns=self.encoder.get_feature_names_out(cols_to_one_hot_encode),
        #     index=processed_df.index
        # )
        # final_data = processed_df.drop(columns=cols_to_one_hot_encode).join(encoded_df)
        final_data = self.preprocessor.transform(processed_df)
        return final_data

    def model_prediction(self, input_data: pd.DataFrame) -> int:
        prediction = int(round(self.model.predict(input_data)[0]))
        return prediction


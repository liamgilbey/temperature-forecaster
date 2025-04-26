import torch
import numpy as np
from datetime import datetime, timedelta
import polars as pl
import data as dt
from dotenv import load_dotenv
import logger

class TemperatureModel:
    def __init__(self, input_data: pl.DataFrame, logger):
        self.input_data = input_data
        self.logger = logger
        self.training_data_x = None
        self.training_data_y = None
        self.model = None

    def create_samples(self, df: pl.DataFrame, input_hours: int = 3):
        self.logger.info("Creating time samples for training")
        x_samples = []
        y_samples = []
        sensor_cols = ["state"]
        
        # Convert to pandas for easy slicing (PyTorch expects numpy arrays)
        pdf = df.to_pandas()
        
        timestamps = pdf['timestamp'].values
        interval_minutes = 60
        input_steps = int(input_hours * 60 / interval_minutes)

        for idx in range(input_steps, len(pdf)):
            current_time = pdf['timestamp'].iloc[idx]
            # Only use times before midnight
            if current_time.time() >= datetime.strptime("00:00", "%H:%M").time() and \
            current_time.time() < datetime.strptime("23:45", "%H:%M").time():

                input_window = pdf.iloc[idx-input_steps:idx]
                future_window = pdf.iloc[idx: ]

                # How many steps until midnight
                minutes_to_midnight = (datetime.combine(current_time.date(), datetime.max.time()) - current_time).seconds // 60
                steps_to_midnight = minutes_to_midnight // interval_minutes

                input_features = input_window[sensor_cols + ["time_sin", "time_cos"]].values
                future_targets = future_window[sensor_cols].values[:steps_to_midnight]

                x_samples.append(input_features)
                y_samples.append(future_targets)

        self.training_data_x = x_samples
        self.training_data_y = y_samples
        return x_samples, y_samples

if __name__ == "__main__":
    load_dotenv()
    log = logger.Logger()

    sensor_data = dt.SensorData(log.logger)
    df = sensor_data.load_csv("history.csv")
    df = sensor_data.add_time_features()

    model = TemperatureModel(df, log.logger) 
    x_samples, y_samples = model.create_samples(df, input_hours=3)
    print(x_samples)
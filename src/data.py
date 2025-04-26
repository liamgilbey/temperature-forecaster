import polars as pl
import numpy as np
import os

class SensorData:
    def __init__(self, data_folder: str = "./data"):
        self.data_folder = data_folder
        self.data = None

    def load_csv(self, file_name: str):
        """
        Load a CSV file from the data folder.
        Args:
            file_name (str): The name of the CSV file to load.
        """
        file_path = os.path.join(self.data_folder, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_name} not found in {self.data_folder}")
        schema_overrides = {
            "entity_id": pl.Utf8,
            "state": pl.Float64,
            "last_changed": pl.Utf8  # Assuming 'last_changed' is a string in the CSV
        }
        self.data = pl.read_csv(file_path, schema_overrides=schema_overrides, null_values=["unavailable"])
        return self.data
    
    def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
        """
        Add time-based features to the DataFrame.
        Args:
            df (pl.DataFrame): The DataFrame to which to add time features.
        Returns:
            pl.DataFrame: The DataFrame with added time features.
        """
        # Ensure the timestamp column is in datetime format
        df = df.with_columns([
            pl.col("last_changed").str.strptime(pl.Datetime).alias("timestamp")
        ])

        # compute the hour and the minute
        df = df.with_columns([
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.minute().alias("minute")
        ])
        # Time as fraction of day
        df = df.with_columns([
            ((pl.col("hour") * 60 + pl.col("minute")) / (24 * 60)).alias("time_frac")
        ])
        # compute the harmonic transformations of time
        df = df.with_columns([
            (pl.col("time_frac") * 2 * np.pi).sin().alias("time_sin"),
            (pl.col("time_frac") * 2 * np.pi).cos().alias("time_cos")
        ])
        return df.drop(["hour", "minute", "time_frac"])
    
if __name__ == "__main__":
    sensor_data = SensorData()
    df = sensor_data.load_csv("history.csv")
    df = SensorData.add_time_features(df)
    print(df.head())
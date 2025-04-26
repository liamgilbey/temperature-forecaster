import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
import polars as pl
import data as dt
from dotenv import load_dotenv
import logger

class TemperatureModel(nn.Module):
    def __init__(self, input_data: pl.DataFrame, input_size, hidden_size, output_size, logger, training_epochs: int = 5):
        super().__init__()
        self.input_data = input_data
        self.logger = logger
        self.training_epochs = training_epochs

        self.samples = None
        
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def create_samples(self, df: pl.DataFrame, input_hours: int = 3, future_steps: int = 3):
        self.logger.info("Creating time samples for training")
        samples = []
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
                future_window = pdf.iloc[idx:idx+future_steps]

                input_features = input_window[sensor_cols + ["time_sin", "time_cos"]].values
                future_targets = future_window[sensor_cols].values

                # Pad future_targets to ensure consistent size
                if len(future_targets) < future_steps:
                    padding = np.zeros((future_steps - len(future_targets), future_targets.shape[1]))
                    future_targets = np.vstack([future_targets, padding])                

                samples.append((input_features, future_targets))

        self.samples = samples
        return samples
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # take last hidden state
        out = self.fc(out)
        return out
    
    def train_model(self):
        self.logger.info("Creating loader object")
        dataset = TempDataset(samples)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        self.logger.info("Creating model")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        self.logger.info("Training model")

        for epoch in range(self.training_epochs):
            total_loss = 0
            for xb, yb in loader:
                preds = model(xb)

                target = yb[:, 0, :]  # only predict first future step for now
                loss = loss_fn(preds, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch} | Loss {total_loss/len(loader)}")

    def predict(self, input_features: np.ndarray):
        """
        Generate predictions for the given input features.
        Args:
            input_features (np.ndarray): Input features of shape (seq_len, input_size).
        Returns:
            np.ndarray: Predicted output.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            predictions = self(input_tensor)  # Forward pass
        return predictions.squeeze(0).numpy()  # Remove batch dimension            
    
class TempDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y    

if __name__ == "__main__":
    load_dotenv()
    log = logger.Logger()

    sensor_data = dt.SensorData(log.logger)
    df = sensor_data.load_csv("history.csv")
    df = sensor_data.add_time_features()

    model = TemperatureModel(df, input_size=3, hidden_size=64, output_size=1, training_epochs=30, logger = log.logger) 
    samples = model.create_samples(df, input_hours=24)

    log.logger.info(f"Number of samples created: {len(samples)}")
    model.train_model()

    # Get predictions
    log.logger.info("Generating predictions")
    all_predictions = []
    all_targets = []
    all_timestamps = []

    for i, (input_features, target) in enumerate(samples):
        pred = model.predict(input_features)
        all_predictions.append(pred.item())  # Ensure it's a float
        all_targets.append(target[0][0].item())  # First future step, first sensor

        # Get the timestamp corresponding to the prediction
        timestamp = df['timestamp'][i + input_features.shape[0] - 1]
        all_timestamps.append(timestamp)

    # Build a Polars DataFrame
    results_df = pl.DataFrame({
        "timestamp": all_timestamps,
        "prediction": all_predictions,
        "actual": all_targets
    })

    # Optional: calculate error
    results_df = results_df.with_columns(
        (pl.col("prediction") - pl.col("actual")).alias("error")
    )

    print(results_df.head())

    # Save to CSV
    results_df.write_csv("predictions.csv")
    print("Saved predictions to predictions.csv ✅")

   
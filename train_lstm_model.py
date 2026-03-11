"""
LSTM Neural Network model for stock price prediction.
Trains on historical Close prices with last year as validation data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging


def load_close_data(csv_file):
    """
    Load Close price data from historical CSV.
    
    Args:
        csv_file (Path): Path to CSV file
        
    Returns:
        pd.DataFrame: DataFrame with Date and Close columns
    """
    try:
        data = pd.read_csv(csv_file)
        
        # Find date column (could be 'Date' or first unnamed column)
        date_col = None
        if "Date" in data.columns:
            date_col = "Date"
        elif data.columns[0].startswith("Unnamed"):
            date_col = data.columns[0]
            data = data.rename(columns={date_col: "Date"})
            date_col = "Date"
        
        if date_col is None or "Close" not in data.columns:
            print(f"Error: {csv_file.name} missing Date or Close column")
            return None
        
        # Convert to datetime and sort
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date", "Close"])
        data = data.sort_values("Date").reset_index(drop=True)
        
        return data[["Date", "Close"]]
    
    except Exception as e:
        print(f"Error loading {csv_file.name}: {e}")
        return None


def split_train_validation(data, validation_days=365):
    """
    Split data into training and validation sets.
    Validation set is the last N days (default 365 = 1 year).
    
    Args:
        data (pd.DataFrame): Data with Date and Close columns
        validation_days (int): Number of days for validation
        
    Returns:
        tuple: (train_data, validation_data)
    """
    split_point = len(data) - validation_days
    if split_point <= 0:
        print(f"Warning: Not enough data. Total records: {len(data)}")
        split_point = int(len(data) * 0.8)  # Use 80/20 split as fallback
    
    train_data = data.iloc[:split_point].copy()
    validation_data = data.iloc[split_point:].copy()
    
    return train_data, validation_data


def create_sequences(data, lookback=60):
    """
    Create sequences for LSTM training.
    
    Args:
        data (np.array): Normalized data
        lookback (int): Number of previous timesteps to use
        
    Returns:
        tuple: (X, y) where X is sequences and y is targets
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    """
    Build LSTM model architecture.
    
    Args:
        input_shape (tuple): Shape of input (lookback, features)
        
    Returns:
        keras.Model: Compiled LSTM model
    """
    model = keras.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(50, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(25),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


def main():
    """Main training function."""
    
    print("=" * 70)
    print("LSTM MODEL TRAINING FOR STOCK PRICE PREDICTION")
    print("=" * 70)
    
    # Configuration
    data_historical_dir = Path("data/historical/")
    data_models_dir = Path("data/models/")
    images_dir = Path("data/models/images/")
    lookback = 60  # Use last 60 days to predict next day
    epochs = 50
    batch_size = 32
     

    if not data_historical_dir.exists():
        print(f"\nError: Historical data directory not found: {data_historical_dir}")
        return
    
    if not data_models_dir.exists():
        data_models_dir.mkdir(parents=True)
        print(f"Created models directory: {data_models_dir}")

    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created images directory: {images_dir}")

    # Find historical CSV files
    csv_files = sorted(data_historical_dir.glob("*_historical_*.csv"))
    if not csv_files:
        print("\nError: No historical CSV files found in data folder.")
        return
    
    for i, csv_file in enumerate(csv_files, 1):

        print(f"  {i}. {csv_file.name}")

        # Configure logging
        logs_dir = data_models_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        log_file = logs_dir / f"{csv_file.stem}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"\nProcessing: {csv_file.name}")
        
        # Load data
        data = load_close_data(csv_file)
        if data is None:
            continue
        
        logger.info(f"Total records: {len(data)}")
        logger.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        logger.info(f"Close price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
        # Normalize data first
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data[['Close']].values)
        data['Close'] = data_scaled
        
        validation_days = int(len(data) * 0.1) 
        # Split into train and validation
        train_data, val_data = split_train_validation(data, validation_days)
        logger.info(f"\nTrain set: {len(train_data)} records ({train_data['Date'].min()} to {train_data['Date'].max()})")
        logger.info(f"Validation set: {len(val_data)} records ({val_data['Date'].min()} to {val_data['Date'].max()})")
        
        train_scaled = train_data[['Close']].values
        val_scaled = val_data[['Close']].values
        
        # Create sequences
        X_train, y_train = create_sequences(train_scaled, lookback)
        X_val, y_val = create_sequences(val_scaled, lookback)
        
        logger.info(f"\nSequence shapes:")
        logger.info(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
        
        if len(X_train) == 0 or len(X_val) == 0:
            logger.error("\nError: Not enough data to create sequences. Need more historical data.")
            continue
        
        # Build model
        logger.info(f"\nBuilding LSTM model...")
        model = build_lstm_model((lookback, 1))
        logger.info(model.summary())
        
        # Train model
        logger.info(f"\nTraining model ({epochs} epochs)...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Make predictions
        logger.info(f"\nMaking predictions...")
        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)
        
        # Inverse transform to original scale
        train_predictions = scaler.inverse_transform(train_predictions)
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
        val_predictions = scaler.inverse_transform(val_predictions)
        y_val_actual = scaler.inverse_transform(y_val.reshape(-1, 1))
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
        train_mae = mean_absolute_error(y_train_actual, train_predictions)
        val_rmse = np.sqrt(mean_squared_error(y_val_actual, val_predictions))
        val_mae = mean_absolute_error(y_val_actual, val_predictions)
        
        logger.info("\n" + "=" * 70)
        logger.info(f"TRAINING RESULTS - {csv_file.name}")
        logger.info("=" * 70)
        logger.info(f"Training RMSE: ${train_rmse:.2f}")
        logger.info(f"Training MAE: ${train_mae:.2f}")
        logger.info(f"Validation RMSE: ${val_rmse:.2f}")
        logger.info(f"Validation MAE: ${val_mae:.2f}")
        
        # Save model
        model_path = data_models_dir / f"{csv_file.stem}_lstm_model.h5"
        model.save(model_path)
        logger.info(f"\nModel saved to: {model_path}")
        
        # Plot results
        logger.info("\nGenerating plots...")
        
        # Plot 1: Training history
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_ylabel('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['mae'], label='Training MAE')
        axes[1].plot(history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_ylabel('MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        

        plt.tight_layout()
        plt.savefig(data_models_dir / f"images/{csv_file.stem}_training_history.png")
        plt.close()
        
        # Plot 2: Predictions vs Actual
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Training predictions
        train_dates = train_data['Date'].iloc[lookback:].values
        axes[0].plot(train_dates, y_train_actual, label='Actual', alpha=0.7)
        axes[0].plot(train_dates, train_predictions, label='Predicted', alpha=0.7)
        axes[0].set_title(f'Training Set - Predictions vs Actual (RMSE: ${train_rmse:.2f})')
        axes[0].set_ylabel('Close Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Validation predictions
        val_dates = val_data['Date'].iloc[lookback:].values
        axes[1].plot(val_dates, y_val_actual, label='Actual', alpha=0.7)
        axes[1].plot(val_dates, val_predictions, label='Predicted', alpha=0.7)
        axes[1].set_title(f'Validation Set - Predictions vs Actual (RMSE: ${val_rmse:.2f})')
        axes[1].set_ylabel('Close Price ($)')
        axes[1].set_xlabel('Date')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(data_models_dir / f"images/{csv_file.stem}_predictions.png")
        plt.close()
        
        logger.info(f"\nCompleted {csv_file.name}!")
    
    print("\n" + "=" * 70)
    print("All files processed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()

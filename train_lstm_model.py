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
    data_dir = Path("data")
    lookback = 60  # Use last 60 days to predict next day
    epochs = 50
    batch_size = 32
    validation_days = 365  # Last year for validation
    
    if not data_dir.exists():
        print("\nError: data folder not found. Run save_advanced_csv.py first.")
        return
    
    # Find historical CSV files
    csv_files = sorted(data_dir.glob("*_historical_*.csv"))
    if not csv_files:
        print("\nError: No historical CSV files found in data folder.")
        return
    
    print(f"\nFound {len(csv_files)} historical file(s):")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file.name}")
    
    # Process first file (or you could add selection logic)
    csv_file = csv_files[0]
    print(f"\nProcessing: {csv_file.name}")
    
    # Load data
    data = load_close_data(csv_file)
    if data is None:
        return
    
    print(f"Total records: {len(data)}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Close price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    # Split into train and validation
    train_data, val_data = split_train_validation(data, validation_days)
    print(f"\nTrain set: {len(train_data)} records ({train_data['Date'].min()} to {train_data['Date'].max()})")
    print(f"Validation set: {len(val_data)} records ({val_data['Date'].min()} to {val_data['Date'].max()})")
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data[['Close']].values)
    val_scaled = scaler.transform(val_data[['Close']].values)
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, lookback)
    X_val, y_val = create_sequences(val_scaled, lookback)
    
    print(f"\nSequence shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    if len(X_train) == 0 or len(X_val) == 0:
        print("\nError: Not enough data to create sequences. Need more historical data.")
        return
    
    # Build model
    print(f"\nBuilding LSTM model...")
    model = build_lstm_model((lookback, 1))
    print(model.summary())
    
    # Train model
    print(f"\nTraining model ({epochs} epochs)...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Make predictions
    print("\nMaking predictions...")
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
    
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    print(f"Training RMSE: ${train_rmse:.2f}")
    print(f"Training MAE: ${train_mae:.2f}")
    print(f"Validation RMSE: ${val_rmse:.2f}")
    print(f"Validation MAE: ${val_mae:.2f}")
    
    # Save model
    model_path = data_dir / f"{csv_file.stem}_lstm_model.h5"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Plot results
    print("\nGenerating plots...")
    
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
    plt.show()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

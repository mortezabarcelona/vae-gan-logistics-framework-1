import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Define preprocessing function
def preprocess_logistics_data(input_filepath, output_dir="data/processed"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    logging.info(f"Loading dataset from {input_filepath}")
    df = pd.read_csv(input_filepath)
    logging.info(f"Dataset loaded with shape {df.shape}")

    # Define column types
    categorical_cols = [
        'day_of_week', 'transport_mode', 'weather_condition',
        'port_status', 'road_incident'
    ]
    numerical_cols = [
        'time_step', 'hour_of_day', 'demand_level', 'volume', 'weight',
        'distance', 'transit_time', 'fuel_price', 'co2_emissions',
        'fuel_cost', 'emission_penalty', 'weather_severity',
        'traffic_congestion', 'driver_fatigue', 'port_congestion',
        'shipment_urgency', 'customer_satisfaction', 'delivery_deadline'
    ]
    # Columns to drop (not useful for training or already encoded elsewhere)
    drop_cols = ['shipment_id', 'origin', 'destination', 'route']

    # 1. Drop unnecessary columns
    logging.info("Dropping unnecessary columns")
    df = df.drop(columns=drop_cols)

    # 2. Encode categorical variables (one-hot encoding)
    logging.info("Encoding categorical variables")
    df_encoded = pd.get_dummies(df, columns=categorical_cols, dtype=float)

    # 3. Normalize numerical columns to [0, 1]
    logging.info("Normalizing numerical columns")
    scaler = MinMaxScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

    # 4. Convert to NumPy array
    logging.info("Converting to NumPy array")
    data_array = df_encoded.to_numpy()
    logging.info(f"Final preprocessed data shape: {data_array.shape}")

    # 5. Split into training and validation sets (80-20 split)
    logging.info("Splitting into training and validation sets")
    train_data, val_data = train_test_split(
        data_array, test_size=0.2, random_state=42, stratify=df['transport_mode']
    )
    logging.info(f"Training set shape: {train_data.shape}")
    logging.info(f"Validation set shape: {val_data.shape}")

    # 6. Save preprocessed data
    timestamp = int(os.path.basename(input_filepath).split('_')[-1].split('.')[0])
    train_path = os.path.join(output_dir, f"train_data_{timestamp}.npy")
    val_path = os.path.join(output_dir, f"val_data_{timestamp}.npy")
    np.save(train_path, train_data)
    np.save(val_path, val_data)
    logging.info(f"Saved training data to {train_path}")
    logging.info(f"Saved validation data to {val_path}")


# Example usage
if __name__ == "__main__":
    preprocess_logistics_data("data/synthetic/logistics_data_baseline_1746285358.csv")
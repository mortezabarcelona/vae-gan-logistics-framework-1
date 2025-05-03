# preprocess.py
# -------------------------------------------------
# Prepares raw or synthetic logistics data for VAE-GAN training
# Tasks: one-hot encode categorical vars, normalize numerics, export ML-ready CSV

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Input/output paths
INPUT_PATH = "data/synthetic/logistics_data_baseline_1746275160.csv"
OUTPUT_PATH = "data/processed/logistics_data_processed.csv"

# Columns to use in model
NUMERIC_FEATURES = ['volume', 'distance_km', 'transit_time', 'cost', 'co2_emissions', 'urgency', 'satisfaction']
CATEGORICAL_FEATURES = ['mode']
DROP_COLUMNS = ['shipment_id', 'origin_city', 'destination_city']  # If present

# Main preprocessing function
def preprocess_data(input_path, output_path):
    if not os.path.exists(input_path):
        logging.error(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    logging.info(f"Loaded dataset with shape: {df.shape}")

    # Drop unnecessary columns if present
    for col in DROP_COLUMNS:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Define transformers
    scaler = MinMaxScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, NUMERIC_FEATURES),
            ('cat', encoder, CATEGORICAL_FEATURES)
        ]
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    processed_array = pipeline.fit_transform(df)

    # Create feature names
    cat_feature_names = list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_feature_names

    processed_df = pd.DataFrame(processed_array, columns=feature_names)
    logging.info(f"Preprocessed dataset shape: {processed_df.shape}")

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    logging.info(f"Saved preprocessed data to: {output_path}")

# Example usage
if __name__ == "__main__":
    preprocess_data(INPUT_PATH, OUTPUT_PATH)

import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define validation function
def validate_logistics_data(filepath):
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return

    df = pd.read_csv(filepath)
    logging.info(f"Loaded dataset: {filepath} with shape {df.shape}")

    # 1. Check for missing values
    missing_report = df.isnull().sum()
    if missing_report.any():
        logging.warning("Missing values detected:")
        logging.warning(f"\n{missing_report[missing_report > 0]}")
    else:
        logging.info("✅ No missing values found.")

    # 2. Check for duplicates
    if df.duplicated().any():
        logging.warning(f"Found {df.duplicated().sum()} duplicated rows.")
    else:
        logging.info("✅ No duplicated rows.")

    # 3. Validate transport mode distribution
    if 'transport_mode' in df.columns:
        mode_distribution = df['transport_mode'].value_counts(normalize=True)
        logging.info("Transport mode distribution:")
        logging.info(f"\n{mode_distribution}")
    else:
        logging.warning("Column 'transport_mode' not found. Skipping mode distribution check.")

    # 4. Check for negative or extreme values
    for col in ['transit_time', 'fuel_cost', 'co2_emissions']:
        if col in df.columns:
            negatives = df[df[col] < 0]
            if not negatives.empty:
                logging.warning(f"{len(negatives)} rows have negative values in {col}.")
            outliers = df[df[col] > df[col].quantile(0.999)]
            if not outliers.empty:
                logging.info(f"{len(outliers)} potential outliers detected in {col}.")
        else:
            logging.warning(f"Column '{col}' not found in dataset.")

    # 5. Check urgency and satisfaction range
    for col in ['shipment_urgency', 'customer_satisfaction']:
        if col in df.columns:
            if not df[col].between(0, 1).all():
                logging.warning(f"Values outside [0,1] found in {col}.")
            else:
                logging.info(f"✅ {col} is within expected range [0,1].")

    logging.info("Validation complete.")

# Example usage:
if __name__ == "__main__":
    validate_logistics_data("data/synthetic/logistics_data_baseline_1746285358.csv")
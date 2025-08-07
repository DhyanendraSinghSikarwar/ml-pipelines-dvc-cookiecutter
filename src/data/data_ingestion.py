import pandas as pd
import numpy as np
import os
import yaml 
import logging

from sklearn.model_selection import train_test_split

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def load_params() -> float:
    try:
        return yaml.safe_load(open('params.yaml', 'r'))
    except FileNotFoundError as e:
        logger.error(f"Params file not found: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading params: {e}")
        return {}

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        logger.error(f"Error reading data from {url}: {e}")
        return pd.DataFrame()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=['tweet_id'], inplace=True)
    final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
    final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
    return final_df

def save_data(data_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    os.makedirs(data_path, exist_ok=True)
    train_df.to_csv(os.path.join(data_path, "train.csv"))
    test_df.to_csv(os.path.join(data_path, "test.csv"))

def main():
    params = load_params()
    test_size = params['data_ingestion']['test_size']
    try:
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    if df.empty:
        logger.warning("DataFrame is empty after loading.")
        return

    try:
        final_df = preprocess_data(df)
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
        return

    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

    data_path = os.path.join("data","raw")
    save_data(data_path, train_data, test_data)

if __name__ == "__main__":
    main()
    # This will execute the main function when the script is run directly.
    # If this script is imported as a module, the main function will not execute automatically.
    # This is a common Python idiom to allow or prevent parts of code from being run when the modules are imported.
import pandas as pd
import numpy as np
import os
import yaml

from sklearn.feature_extraction.text import CountVectorizer

def load_params():
    return yaml.safe_load(open('params.yaml', 'r'))


def load_data():
    # fetch the data from data/processed
    try:
        train_data = pd.read_csv('./data/processed/train_processed.csv')
        test_data = pd.read_csv('./data/processed/test_processed.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error: {e}")
        return pd.DataFrame(), pd.DataFrame()
    # Ensure the data is not empty
    if train_data.empty or test_data.empty:
        print("Data files are empty.")
        return pd.DataFrame(), pd.DataFrame()
    return train_data, test_data


def preprocess_features(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    # Apply BOW
    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values

    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values

    train_data.fillna('', inplace=True)
    test_data.fillna('', inplace=True)
    return X_train, y_train, X_test, y_test


def apply_bow(X_train: np.ndarray, X_test: np.ndarray, max_features: int) -> tuple:
    # Apply Bag of Words
    vectorizer = CountVectorizer(max_features=max_features)

    # Fit and transform the training data, transform the test data
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    return X_train_bow, X_test_bow, vectorizer


def convert_to_dataframe(X_train_bow: np.ndarray, X_test_bow: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, vectorizer: CountVectorizer) -> tuple:
    # Convert to DataFrame
    train_df = pd.DataFrame(X_train_bow.toarray(), columns=vectorizer.get_feature_names_out())
    train_df['label'] = y_train
    test_df = pd.DataFrame(X_test_bow.toarray(), columns=vectorizer.get_feature_names_out())
    test_df['label'] = y_test
    return train_df, test_df


# store the data inside data/features
def save_data(data_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    os.makedirs(data_path, exist_ok=True)
    try:
        train_df.to_csv(os.path.join(data_path, 'train_bow.csv'), index=False)
        test_df.to_csv(os.path.join(data_path, 'test_bow.csv'), index=False)
    except Exception as e:
        print(f"Error saving data: {e}")


def main():
    params = load_params()
    try:
        max_features = params['feature_engineering']['max_features']
    except KeyError as e:
        print(f"Parameter missing in params.yaml: {e}")
        return

    # fetch the data from data/processed
    train_data, test_data = load_data()
    
    # Preprocess features
    X_train, y_train, X_test, y_test = preprocess_features(train_data, test_data)

    # Apply BOW
    X_train_bow, X_test_bow, vectorizer = apply_bow(X_train, X_test, max_features)

    # Convert to DataFrame
    train_df, test_df = convert_to_dataframe(X_train_bow, X_test_bow, y_train, y_test, vectorizer)
    
    data_path = os.path.join('data', 'features')
    # Save the processed data
    save_data(data_path, train_df, test_df)

if __name__ == "__main__":
    main()

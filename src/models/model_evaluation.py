import numpy as np
import pandas as pd
import pickle
import json

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from src.utils import setup_logging

logger = setup_logging(__name__)

def load_model():
    # Load the model
    return pickle.load(open('models/model.pkl', 'rb'))


def load_data():
    # fetch the data from data/features
    try:
        test_data = pd.read_csv('./data/features/test_tfidf.csv')
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return pd.DataFrame()
    # Ensure the data is not empty
    if test_data.empty:
        logger.warning("Data file is empty.")
        return pd.DataFrame()
    logger.info("Data loaded successfully.")
    return test_data


def prepare_data(test_data: pd.DataFrame) -> tuple:
    # Prepare the data for testing
    X_test = test_data.iloc[:, :-1].values  # All columns except the last one
    y_test = test_data.iloc[:, -1].values   # Last column is the label
    return X_test, y_test


def make_predictions(clf: GradientBoostingClassifier, X_test: np.ndarray) -> tuple:
    # Make predictions
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        logger.info("Predictions made successfully.")
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return np.array([]), np.array([])

    return y_pred, y_pred_proba


def evaluate_model(y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> tuple:
    # Calculate evaluation metrics
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        logger.info("Model evaluation completed successfully.")
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return 0.0, 0.0, 0.0, 0.0
    return accuracy, precision, recall, auc


def save_metrics(accuracy: float, precision: float, recall: float, auc: float) -> None:
    # Save the metrics to a JSON file
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }
    
    with open('reports/metrics.json', 'w') as file:
        json.dump(metrics_dict, file, indent=4)
    logger.info("Metrics saved successfully.")


def main():
    # Load the model
    clf = load_model()
    
    # Load the data
    test_data = load_data()
    
    # Prepare the data
    X_test, y_test = prepare_data(test_data)
    
    # Make predictions
    y_pred, y_pred_proba = make_predictions(clf, X_test)
    
    # Evaluate the model
    accuracy, precision, recall, auc = evaluate_model(y_test, y_pred, y_pred_proba)
    
    # Save the metrics
    save_metrics(accuracy, precision, recall, auc)
    logger.info("Model evaluation completed successfully.")

if __name__ == "__main__":
    main()

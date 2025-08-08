import pandas as pd
import numpy as np

import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from src.utils import setup_logging

logger = setup_logging(__name__)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

def load_data():
    # fetch the data from data/raw
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.info("Data loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()
    return train_data, test_data

# transform the text data
def lemmatization(text: str) -> str:
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text: str) -> str:
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text: str) -> str:

    text = text.split()

    text = [y.lower() for y in text]

    return " " .join(text)

def removing_punctuations(text: str) -> str:
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text: str) -> str:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.content = df.content.apply(lambda content: lower_case(content))
        df.content = df.content.apply(lambda content: remove_stop_words(content))
        df.content = df.content.apply(lambda content: removing_numbers(content))
        df.content = df.content.apply(lambda content: removing_punctuations(content))
        df.content = df.content.apply(lambda content: removing_urls(content))
        df.content = df.content.apply(lambda content: lemmatization(content))
    except Exception as e:
        logger.error(f"Error normalizing text: {e}")
    return df

def save_processed_data(data_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    os.makedirs(data_path, exist_ok=True)
    train_df.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
    test_df.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
    logger.info("Processed data saved successfully.")

def main():
    # Load the data
    train_data, test_data = load_data()

    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)

    # store the preprocessed data at data/processed
    data_path = os.path.join('./data/processed')
    try:
        save_processed_data(data_path, train_processed_data, test_processed_data)
    except Exception as e:
       logger.error(f"Error saving processed data: {e}")

if __name__ == "__main__":
    main()

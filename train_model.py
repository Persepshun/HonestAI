import os
import pickle
import random
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Assuming fetch_gdelt_data and preprocess_gdelt_data are in fetch_news.py and preprocess_data.py
from fetch_news import fetch_gdelt_data
from preprocess_data import preprocess_gdelt_data

# Set up local directories for saving results
MODEL_DIR = "saved_models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def fetch_and_preprocess_data(query="technology", timespan="1d", max_records=100):
    """
    Fetches data from GDELT and preprocesses it for training.
    """
    # Fetch data from GDELT API
    raw_data = fetch_gdelt_data(query=query, timespan=timespan, max_records=max_records)
    if raw_data.empty:
        print("No data fetched from GDELT.")
        return None

    # Preprocess data
    processed_data = preprocess_gdelt_data(raw_data)
    return processed_data

def train_model(data):
    """
    Trains a Logistic Regression model on preprocessed data.
    Saves the trained model and outputs evaluation metrics.
    """
    # Prepare features (X) and target (y) from preprocessed data
    X = data["cleaned_content"].apply(len).values.reshape(-1, 1)  # Example feature: length of content
    y = data["target"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    # Create a unique filename using the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"gdelt_model_{timestamp}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)

    # Randomize file size between 700 and 1500 bytes by adding filler data
    target_size = random.randint(700, 1500)  # Random size between 700 and 1500 bytes
    filler_size = max(0, target_size - len(pickle.dumps(model)))  # Calculate required filler size
    filler_data = os.urandom(filler_size)  # Generate random bytes for filler

    # Save the model with filler data to reach target size
    model_data = {
        'model': model,
        'filler': filler_data
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {model_path} with size {target_size} bytes (approx).")

    # Save results
    results_path = os.path.join(RESULTS_DIR, f"model_results_{timestamp}.txt")
    with open(results_path, "w") as f:
        f.write(f"Model Accuracy: {accuracy}\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Results saved to {results_path}")

def main():
    # Fetch and preprocess data
    data = fetch_and_preprocess_data(query="technology", timespan="1d", max_records=100)
    if data is None:
        print("No data available for training.")
        return

    # Train and evaluate the model
    train_model(data)

if __name__ == "__main__":
    main()

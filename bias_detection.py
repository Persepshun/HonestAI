# bias_detection.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from fairlearn.metrics import MetricFrame, demographic_parity_difference
import matplotlib.pyplot as plt

# Function to load the Adult Income dataset
def load_dataset():
    """
    Load and clean the Adult Income dataset for testing model fairness.
    
    Returns:
        data (DataFrame): Cleaned dataset with columns for features and target.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
               'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
               'hours_per_week', 'native_country', 'income']
    data = pd.read_csv(url, names=columns, na_values=' ?')
    data.dropna(inplace=True)
    return data

# Function to preprocess the data and prepare it for training

def preprocess_data(data):
    """
    Process the input DataFrame, which should contain web page content.
    This version is adjusted for generic content without requiring an "income" column.
    
    Args:
        data (DataFrame): DataFrame with a 'content' column containing text content from a URL.
    
    Returns:
        X (DataFrame): Feature data (processed content).
        y (Series): Dummy target variable, set to zeros for compatibility.
    """
    # Ensure 'content' column exists in the data
    if 'content' not in data.columns:
        raise ValueError("Data does not contain a 'content' column.")
    
    # For simplicity, we'll treat each character count in the content as a feature (basic example)
    # This is just to create dummy features for compatibility with the bias and ethical analysis
    data['char_count'] = data['content'].apply(len)
    data['word_count'] = data['content'].apply(lambda x: len(x.split()))
    
    X = data[['char_count', 'word_count']]
    y = pd.Series([0] * len(X))  # Dummy target variable, set to zeroes
    return X, y


# Train, predict, and calculate accuracy of a logistic regression model
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Train a logistic regression model and evaluate its accuracy.
    
    Returns:
        y_pred (array): Predictions on test data.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return y_pred

# Fairness evaluation with Fairlearn's Demographic Parity Difference
def evaluate_fairness(y_test, y_pred, sensitive_feature):
    """
    Calculate demographic parity difference for sensitive features.
    
    Args:
        y_test (Series): True labels for test set.
        y_pred (array): Predictions on test data.
        sensitive_feature (DataFrame): Dataframe with sensitive feature columns.
        
    Returns:
        dp_difference, dp_difference_sex (float): Demographic parity difference.
    """
    dp_difference = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_feature['race_ White'])
    dp_difference_sex = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_feature['sex_ Male'])
    print(f"Demographic Parity Difference (Race): {dp_difference:.3f}")
    print(f"Demographic Parity Difference (Sex): {dp_difference_sex:.3f}")
    return dp_difference, dp_difference_sex

# Additional AIF360 metrics
def evaluate_aif360_metrics(X_test, y_test):
    """
    Calculate AIF360 fairness metrics: disparate impact and statistical parity difference.
    
    Returns:
        disparate_impact, statistical_parity (float): Calculated fairness metrics.
    """
    privileged_groups = [{'race_ White': 1}]
    unprivileged_groups = [{'race_ White': 0}]
    aif360_data = BinaryLabelDataset(
        df=pd.concat([X_test, y_test], axis=1),
        label_names=['income'],
        protected_attribute_names=['race_ White'],
        privileged_protected_attributes=privileged_groups,
        unprivileged_protected_attributes=unprivileged_groups
    )
    metric = BinaryLabelDatasetMetric(aif360_data, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
    disparate_impact = metric.disparate_impact()
    statistical_parity = metric.statistical_parity_difference()
    print(f"Disparate Impact (Race): {disparate_impact:.3f}")
    print(f"Statistical Parity Difference (Race): {statistical_parity:.3f}")
    return disparate_impact, statistical_parity

# Visualization for fairness metrics
def plot_metrics(metrics):
    """
    Plot fairness metrics for model evaluation.
    
    Args:
        metrics (dict): Dictionary of fairness metrics and values.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.ylabel("Metric Value")
    plt.title("Fairness Metrics for Model Predictions")
    plt.xticks(rotation=45, ha="right")
    plt.show()

    import pandas as pd  # Data manipulation and analysis library
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference  # Fairlearn metrics for bias detection
from aif360.metrics import BinaryLabelDatasetMetric  # Fairness evaluation from AI Fairness 360
from aif360.datasets import BinaryLabelDataset  # Data structuring for fairness checks

def preprocess_data(data):
    """
    Prepares data for model training by calculating character and word count features,
    which will serve as basic features for the model. It also assigns labels as a 'target' variable.
    """
    data['char_count'] = data['content'].apply(len)  # Number of characters in content
    data['word_count'] = data['content'].apply(lambda x: len(x.split()))  # Number of words in content
    X = data[['char_count', 'word_count']]  # Features for the model
    y = data['target']  # Target labels for classification
    return X, y

def evaluate_fairness(y_true, y_pred, sensitive_feature):
    """
    Evaluates the model's predictions for fairness using demographic parity difference.
    Sensitive features are specified to examine if fairness differs across demographics.
    """
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    return dp_diff, eo_diff

def evaluate_aif360_metrics(X, y):
    """
    Uses AI Fairness 360 metrics to calculate fairness measures.
    Assumes the target variable has binary labels, making it suitable for BinaryLabelDataset.
    """
    # Convert data into a BinaryLabelDataset format for AI Fairness 360
    dataset = BinaryLabelDataset(df=pd.DataFrame(X), label_names=['target'], protected_attribute_names=['char_count'])
    metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{'char_count': 0}], privileged_groups=[{'char_count': 1}])
    
    # Fairness evaluation metrics
    disparate_impact = metric.disparate_impact()
    statistical_parity = metric.statistical_parity_difference()
    return disparate_impact, statistical_parity

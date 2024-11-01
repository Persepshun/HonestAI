
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
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
               'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
               'hours_per_week', 'native_country', 'income']
    data = pd.read_csv(url, names=columns, na_values=' ?')
    data.dropna(inplace=True)
    return data

# Function to preprocess the data and prepare it for training
def preprocess_data(data):
    data['income'] = data['income'].apply(lambda x: 1 if x == ' >50K' else 0)
    data = pd.get_dummies(data, drop_first=True)
    X = data.drop('income', axis=1)
    y = data['income']
    return X, y

# Split, train, and evaluate model
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return y_pred

# Fairness evaluation
def evaluate_fairness(y_test, y_pred, sensitive_feature):
    dp_difference = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_feature['race_ White'])
    dp_difference_sex = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_feature['sex_ Male'])
    print(f"Demographic Parity Difference (Race): {dp_difference:.3f}")
    print(f"Demographic Parity Difference (Sex): {dp_difference_sex:.3f}")
    return dp_difference, dp_difference_sex

# Additional AIF360 metrics
def evaluate_aif360_metrics(X_test, y_test):
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

# Visualization
def plot_metrics(metrics):
    plt.figure(figsize=(10, 5))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.ylabel("Metric Value")
    plt.title("Fairness Metrics for Model Predictions")
    plt.xticks(rotation=45, ha="right")
    plt.show()

# Main execution
if __name__ == "__main__":
    data = load_dataset()
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    sensitive_feature = X_test[['race_ White', 'sex_ Male']]
    dp_difference, dp_difference_sex = evaluate_fairness(y_test, y_pred, sensitive_feature)
    disparate_impact, statistical_parity = evaluate_aif360_metrics(X_test, y_test)
    metrics = {
        "Demographic Parity Difference (Race)": dp_difference,
        "Demographic Parity Difference (Sex)": dp_difference_sex,
        "Disparate Impact (Race)": disparate_impact,
        "Statistical Parity Difference (Race)": statistical_parity
    }
    plot_metrics(metrics)

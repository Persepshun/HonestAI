
# run_toolkit.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Import modules
import bias_detection
import ethical_screening

# Load and preprocess the dataset
data = bias_detection.load_dataset()
X, y = bias_detection.preprocess_data(data)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Run Bias Detection
y_pred = model.predict(X_test)
sensitive_feature = X_test[['race_ White', 'sex_ Male']]
dp_difference, dp_difference_sex = bias_detection.evaluate_fairness(y_test, y_pred, sensitive_feature)
disparate_impact, statistical_parity = bias_detection.evaluate_aif360_metrics(X_test, y_test)

# Collect metrics for visualization
metrics = {
    "Demographic Parity Difference (Race)": dp_difference,
    "Demographic Parity Difference (Sex)": dp_difference_sex,
    "Disparate Impact (Race)": disparate_impact,
    "Statistical Parity Difference (Race)": statistical_parity
}
bias_detection.plot_metrics(metrics)

# Run Ethical Screening
ethical_screening.ethical_checklist(X_test, model)

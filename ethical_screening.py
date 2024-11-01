# ethical_screening.py

import lime.lime_tabular
import shap
import numpy as np
import pandas as pd

# Function to run LIME for model explainability
def lime_explain(X_train, X_test, model):
    """
    Uses LIME (Local Interpretable Model-agnostic Explanations) to explain model predictions.
    Provides an interpretability layer by explaining why certain predictions were made.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        mode='classification'
    )
    # Explain a sample prediction
    explanation = explainer.explain_instance(data_row=X_test.iloc[0], predict_fn=model.predict_proba)
    explanation.show_in_notebook()

# Function to run SHAP for feature importance
def shap_explain(X_test, model):
    """
    Uses SHAP (SHapley Additive exPlanations) for understanding feature importance in model predictions.
    Generates a summary plot for interpretability.
    """
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)

# Privacy check function
def privacy_check(features):
    """
    Checks for potential privacy violations by identifying sensitive features within the dataset.
    Flags any features that may contain sensitive data.
    """
    sensitive_features = ['name', 'SSN', 'email', 'address']
    violations = [feature for feature in features if feature in sensitive_features]
    if violations:
        print(f"Privacy Check Warning: Sensitive features found - {violations}")
    else:
        print("Privacy Check Passed: No sensitive features found.")

# Ethical checklist evaluation
def ethical_checklist(X, model):
    """
    Executes an ethical checklist, incorporating explainability, privacy, and simplicity checks.
    Provides warnings if the model or data does not meet ethical standards.
    """
    print("Running Ethical Checklist...")

    # Explainability with LIME
    lime_explain(X_train=X, X_test=X, model=model)

    # Explainability with SHAP
    shap_explain(X_test=X, model=model)

    # Privacy Check
    privacy_check(features=X.columns)

    print("Ethical Checklist Complete.")

def check_privacy_risks(data):
    """
    Evaluates privacy risks by examining if sensitive data (e.g., PII) may be present.
    Flags any content that might contain names or contact info.
    """
    sensitive_keywords = ["name", "address", "phone", "email", "social security", "credit card"]
    for keyword in sensitive_keywords:
        if keyword in data.lower():
            print(f"Privacy Alert: Potential sensitive information found: '{keyword}'")
    print("Privacy risk assessment complete.")

def evaluate_model_transparency(model):
    """
    Assesses model transparency by checking if the model is interpretable and straightforward to explain.
    Linear models and simple trees are considered interpretable.
    """
    if hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'):
        print("Model transparency: The model is interpretable with accessible feature weights.")
    else:
        print("Model transparency warning: The model may not be easily interpretable.")

def check_bias_in_data(X, sensitive_features):
    """
    Examines dataset for potential bias by calculating the representation of sensitive groups.
    Provides warnings if the data is imbalanced in terms of sensitive features.
    """
    for feature in sensitive_features:
        unique_counts = X[feature].value_counts()
        print(f"Bias Check - {feature}:")
        for value, count in unique_counts.items():
            print(f"  Value {value} appears {count} times.")
        if unique_counts.min() / unique_counts.max() < 0.5:
            print(f"Warning: Potential bias detected in {feature}, underrepresented values.")
    print("Bias check in data complete.")

def assess_fairness_metrics(y_true, y_pred, sensitive_feature):
    """
    Calculates fairness metrics like demographic parity and equalized odds based on the sensitive feature.
    Outputs the fairness metrics for analysis and documentation.
    """
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    print(f"Fairness Metrics:")
    print(f"  Demographic Parity Difference: {dp_diff}")
    print(f"  Equalized Odds Difference: {eo_diff}")
    return dp_diff, eo_diff

def document_model_decision_process(model):
    """
    Documents the decision-making process of the model by listing important features.
    Helps in understanding and auditing the model for transparency and accountability.
    """
    if hasattr(model, 'coef_'):
        print("Documenting model decision process based on coefficients:")
        for idx, coef in enumerate(model.coef_[0]):
            print(f"  Feature {idx}: Weight {coef}")
    elif hasattr(model, 'feature_importances_'):
        print("Documenting model decision process based on feature importances:")
        for idx, importance in enumerate(model.feature_importances_):
            print(f"  Feature {idx}: Importance {importance}")
    else:
        print("Warning: Model decision process is not easily documented.")
    print("Model decision process documentation complete.")

def assess_accountability_measures(model):
    """
    Reviews accountability measures by verifying that the model and code are documented and traceable.
    Encourages practices like code versioning and result reproducibility.
    """
    print("Accountability check:")
    print("Ensure model versioning and maintain detailed documentation.")
    if hasattr(model, 'random_state'):
        print("Model has a fixed random state for reproducibility.")
    else:
        print("Warning: Model may not be fully reproducible without a fixed random state.")
    print("Accountability assessment complete.")

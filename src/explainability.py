import shap
import pandas as pd

def generate_shap_summary(model, X_train):
    """
    Generates SHAP summary values for global feature importance.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(shap_values, X_train)

    return explainer, shap_values


def explain_single_prediction(model, X_sample):
    """
    Explains prediction for a single customer.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_sample.iloc[0],
            feature_names=X_sample.columns
        )
    )
from src.preprocessing import load_data, split_data
from src.feature_engineering import add_engineered_features
from src.train_model import train_xgboost, evaluate_model, save_model
from src.explainability import generate_shap_summary, explain_single_prediction

def main():

    df = load_data("data/UCI_Credit_Data.csv")
    df = add_engineered_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_xgboost(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)

    # SHAP Global Explanation
    generate_shap_summary(model, X_train)

    # SHAP Single Customer Explanation
    explain_single_prediction(model, X_test.iloc[[0]])
   

if __name__ == "__main__":
    main()
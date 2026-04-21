import pandas as pd
import joblib

model = joblib.load("models/best_model.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")

X_test = pd.read_csv("data/train_test/X_test.csv")

Xt = preprocessor.transform(X_test)
preds = model.predict(Xt)
probas = model.predict_proba(Xt)[:, 1]

result = X_test.copy()
result["pred_churn"] = preds
result["pred_proba"] = probas
result["distance_to_06"] = (result["pred_proba"] - 0.60).abs()

best_row = result.sort_values("distance_to_06").head(10)

print(best_row[[
    "pred_churn",
    "pred_proba",
    "Frequency",
    "MonetaryTotal",
    "CustomerTenureDays",
    "UniqueProducts",
    "ReturnRatio",
    "Age",
    "SupportTicketsCount",
    "SatisfactionScore",
    "Gender",
    "Country"
]])
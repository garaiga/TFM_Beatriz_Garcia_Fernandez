import joblib

# Guarda el modelo optimizado en un archivo
joblib.dump(best_xgb, "best_xgb_model.pkl")

# Guarda el escalador en un archivo
joblib.dump(scaler, "scaler.pkl")

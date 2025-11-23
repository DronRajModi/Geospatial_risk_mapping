import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# CONFIGURATION
 
ARTIFACTS_DIR = "artifacts"
DATA_FILE = os.path.join(ARTIFACTS_DIR, "data.csv")
IMPUTED_FEATURES_FILE = os.path.join(ARTIFACTS_DIR, "imputed_features.csv")
GNN_EMBED_FILE = os.path.join(ARTIFACTS_DIR, "gnn_embeddings.csv")

PREPROCESSOR_FILE = os.path.join(ARTIFACTS_DIR, "final_preprocessor.pkl")
SCALER_FILE = os.path.join(ARTIFACTS_DIR, "final_scaler.pkl")
MODEL_FILE = os.path.join(ARTIFACTS_DIR, "model_rf.pkl")
LABEL_ENCODER_FILE = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")


 
# LOAD ARTIFACTS
 
def load_artifacts():
    print("[INFO] Loading artifacts...")

    try:
        df_data = pd.read_csv(DATA_FILE)
        df_imputed = pd.read_csv(IMPUTED_FEATURES_FILE)
        df_gnn = pd.read_csv(GNN_EMBED_FILE)
        print("[INFO] Data, imputed features, and GNN embeddings loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load data files: {e}")

    try:
        preprocessor = joblib.load(PREPROCESSOR_FILE)
        scaler = joblib.load(SCALER_FILE)
        model = joblib.load(MODEL_FILE)
        label_encoder = joblib.load(LABEL_ENCODER_FILE)
        print("[INFO] Model artifacts loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load model artifacts: {e}")

    return df_data, df_imputed, df_gnn, preprocessor, scaler, model, label_encoder


 
# PREPARE ENVIRONMENT LOOKUP
 
def prepare_environment_lookup(df_data, df_imputed, df_gnn):
    for df in [df_data, df_imputed, df_gnn]:
        df.columns = [c.strip() for c in df.columns]
        if "district" in df.columns and "District" not in df.columns:
            df.rename(columns={"district": "District"}, inplace=True)

    if "District" not in df_data.columns and "District" not in df_imputed.columns and "District" not in df_gnn.columns:
        raise ValueError("[ERROR] 'District' column missing in all files!")

    df_imputed_gnn = df_imputed.merge(df_gnn, on="District", how="left")
    df_imputed_gnn = df_imputed_gnn.drop_duplicates(subset=["District"], keep="first").set_index("District", drop=False)

    df_data_unique = df_data.drop_duplicates(subset=["District"], keep="first").set_index("District", drop=False)

    primary_districts = set(df_data_unique.index.astype(str).str.strip().str.lower())
    to_add = [d for d in df_imputed_gnn.index if d.strip().lower() not in primary_districts]

    df_env = pd.concat([df_data_unique, df_imputed_gnn.loc[to_add]], axis=0, ignore_index=True)
    df_env["District"] = df_env["District"].astype(str).str.strip()
    df_env = df_env.reset_index(drop=True)

    essential_env = ["NO2", "SO2", "PM2.5", "NDVI", "Population_Density", "Sleep_Hours"]
    for col in essential_env:
        if col not in df_env.columns:
            df_env[col] = 0
            print(f"[WARN] '{col}' not found in environment lookup. Added with default 0.")

    print(f"[INFO] Environment lookup ready. Total districts: {df_env['District'].nunique()}")
    return df_data_unique, df_env


 
# PREDICTION FUNCTION
 
def predict_disease_for_person(person_details, preprocessor, scaler, model, label_encoder, df_data_unique, df_env):
    district = person_details.get("District", "").strip()
    if not district:
        raise ValueError("Please provide a 'District' in person_details.")

    # Find matching district data
    match = df_data_unique[df_data_unique["District"].str.strip().str.lower() == district.lower()]
    if match.empty:
        match = df_env[df_env["District"].str.strip().str.lower() == district.lower()]
    if match.empty:
        raise ValueError(f"[ERROR] District '{district}' not found in available data sources.")

    district_row = match.iloc[0].to_dict()
    input_data = {**person_details, **district_row}
    df_input = pd.DataFrame([input_data])

    # --- Decide which preprocessing to apply ---
    print("\n[INFO] Model expects", model.n_features_in_, "features.")
    print("[INFO] Input has", df_input.shape[1], "columns.")
    print("[INFO] Preprocessor input feature count:", len(preprocessor.feature_names_in_))

    # Step 1: Align columns to preprocessor if feature count matches
    if model.n_features_in_ == len(preprocessor.feature_names_in_):
        print("[INFO] Applying preprocessing + scaling before prediction...")
        # Align columns
        feature_cols = list(preprocessor.feature_names_in_)
        for col in feature_cols:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[feature_cols]

        X_processed = preprocessor.transform(df_input)
        X_scaled = scaler.transform(X_processed)
        X_final = X_scaled

    # Step 2: If model expects fewer features, skip preprocessing
    else:
        print("[INFO] Model trained on preprocessed/scaled features already — skipping preprocessing.")
        # Keep only numeric columns
        df_num = df_input.select_dtypes(include=[np.number]).fillna(0)
        # Align to model’s expected input feature count
        if df_num.shape[1] > model.n_features_in_:
            df_num = df_num.iloc[:, :model.n_features_in_]
        elif df_num.shape[1] < model.n_features_in_:
            for i in range(model.n_features_in_ - df_num.shape[1]):
                df_num[f"extra_{i}"] = 0
        X_final = df_num.values

    # --- Predict disease ---
    try:
        y_pred = model.predict(X_final)
        y_prob = np.max(model.predict_proba(X_final))
        disease_name = label_encoder.inverse_transform(y_pred)[0]
        return {"Disease": disease_name, "Confidence": round(float(y_prob), 3)}
    except Exception as e:
        raise RuntimeError(f"[ERROR] Prediction failed: {e}")



 
# MAIN EXECUTION (for testing)
 
if __name__ == "__main__":
    df_data, df_imputed, df_gnn, preprocessor, scaler, model, label_encoder = load_artifacts()
    df_data_unique, df_env = prepare_environment_lookup(df_data, df_imputed, df_gnn)

    print("\n[INFO] All artifacts loaded successfully.")
    print(f"[INFO] Final merged dataset shape: {df_env.shape}")

    person_details = {
        "District": "Alwar",
        "Age": 5,
        "Gender": "Male",
        "BMI": 28.5,
        "Smoking": "No",
        "Alcohol": "No",
        "Physical_Activity": "Moderate",
        "Sleep_Hours": 7
    }

    try:
        predicted_disease = predict_disease_for_person(
            person_details,
            preprocessor, scaler, model, label_encoder,
            df_data_unique, df_env
        )
        print(f"\nPredicted Disease for {person_details['District']}: {predicted_disease}")
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")

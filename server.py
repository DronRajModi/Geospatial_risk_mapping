import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
import traceback 

# --- 1. SETUP FLASK APP ---
app = Flask(__name__)
CORS(app) 
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- 2. LOAD ALL ARTIFACTS ---
try:
    artifacts_dir = "artifacts"
    model = joblib.load(os.path.join(artifacts_dir, "model_final_spatial_rf.pkl"))
    preprocessor = joblib.load(os.path.join(artifacts_dir, "final_preprocessor.pkl"))
    scaler = joblib.load(os.path.join(artifacts_dir, "final_scaler.pkl"))
    label_encoder = joblib.load(os.path.join(artifacts_dir, "label_encoder.pkl"))

    df_gnn = pd.read_csv(os.path.join(artifacts_dir, "gnn_embeddings.csv"))
    df_gnn['District'] = df_gnn['District'].str.strip().str.title()
    df_gnn = df_gnn.set_index('District')
    
    # --- THIS IS THE FIX ---
    # Load 'data.csv' which has ALL columns (categorical and numerical)
    # 'imputed_features.csv' was the wrong file.
    df_base_data = pd.read_csv(os.path.join(artifacts_dir, "data.csv"))
    df_base_data['District'] = df_base_data['District'].str.strip().str.title()
    # Get the *first* row for each district to use as a template
    df_district_templates = df_base_data.drop_duplicates(subset=['District']).set_index('District')
    # --- END OF FIX ---

    feature_sets = [cols for name, trans, cols in preprocessor.transformers_]
    EXPECTED_COLUMNS = [col for subset in feature_sets for col in subset]

    # Create a synthetic population to test against
    SYNTHETIC_POPULATION_TEMPLATE = []
    ages = [25, 35, 45, 55, 65]
    sexes = ["Male", "Female"]
    for age in ages:
        for sex in sexes:
            SYNTHETIC_POPULATION_TEMPLATE.append({"Age": age, "Sex": sex})
    
    print(f"✅ Server is ready. Loaded {len(df_district_templates)} district templates.")
    print(f"✅ Synthetic population size: {len(SYNTHETIC_POPULATION_TEMPLATE)}")

except FileNotFoundError as e:
    print(f"❌ ERROR: Missing artifact file. Server cannot start. {e}")
    model = None
except Exception as e:
    print(f"❌ ERROR loading artifacts: {e}")
    traceback.print_exc()
    model = None


# --- 3. GOING BACK TO THE WORKING HEATMAP ENDPOINT ---
@app.route("/get_risk_heatmap", methods=["GET"])
def get_risk_heatmap():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    target_disease = request.args.get("disease") # e.g., "CVD"
    if not target_disease:
        return jsonify({"error": "No disease specified"}), 400
    
    # Force target disease to UPPERCASE to match React
    target_disease_upper = target_disease.upper()
    
    # Check if the *original* cased disease name is in the encoder
    if target_disease not in label_encoder.classes_ and target_disease_upper not in label_encoder.classes_:
         return jsonify({"error": f"Disease '{target_disease}' not recognized."}), 400

    print(f"Generating heatmap for: {target_disease}")
    
    try:
        risk_data = {}
        
        # --- FIX: Loop over common districts ---
        imputed_districts = set(df_district_templates.index.unique())
        gnn_districts = set(df_gnn.index.unique())
        all_districts = list(imputed_districts.intersection(gnn_districts))
        print(f"Found {len(all_districts)} common districts to process.")
        # --- END OF FIX ---

        for district in all_districts:
            try:
                # 1. Get base template row (this has 'Gender', 'Tobacco_Use', etc.)
                features_row_template = df_district_templates.loc[[district]]
                # 2. Get GNN features
                gnn_features = df_gnn.loc[[district]]
                
                # 3. Merge them
                full_features_template = pd.merge(features_row_template, gnn_features, left_index=True, right_index=True)

                # 4. Create the full synthetic population for this district
                pop_df_list = []
                for person_template in SYNTHETIC_POPULATION_TEMPLATE:
                    person_row = full_features_template.copy()
                    # Overwrite Age and Sex for the synthetic person
                    person_row["Age"] = person_template["Age"]
                    person_row["Sex"] = person_template["Sex"]
                    pop_df_list.append(person_row)
                
                final_pop_df = pd.concat(pop_df_list)
                final_input_df = final_pop_df[EXPECTED_COLUMNS] # This will no longer fail
                
                # 5. Preprocess and predict
                processed_data = preprocessor.transform(final_input_df)
                scaled_data = scaler.transform(processed_data)
                
                # Get probabilities
                predictions_proba = model.predict_proba(scaled_data)
                
                # 6. Find the risk for the target disease
                target_disease_index = np.where(label_encoder.classes_ == target_disease)[0][0]
                avg_disease_risk = predictions_proba[:, target_disease_index].mean()
                
                risk_percentage = round(avg_disease_risk * 100, 2)

                # Normalize district key to UPPERCASE for matching
                district_key = district.split(' (')[0].strip().upper()
                risk_data[district_key] = risk_percentage

            except KeyError as ke:
                # print(f"Skipping {district}. Missing key: {ke}")
                continue 
            except Exception as e:
                # print(f"Failed {district}. Error: {e}")
                continue

        print(f"Heatmap generation complete. Populated data for {len(risk_data)} districts.")
        return jsonify(risk_data)

    except Exception as e:
        print(f"Heatmap Error: {e}")
        return jsonify({"error": str(e)}), 500


# --- 4. YOUR ORIGINAL PREDICT ENDPOINT (for personal risk) ---
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500
    try:
        data = request.json
        district = data.get("district").strip().title()

        # --- FIX: Use df_district_templates ---
        if district not in df_district_templates.index:
            return jsonify({"error": f"District '{district}' not found in template data."}), 404
        
        features_row = df_district_templates.loc[[district]].copy()
        # --- END OF FIX ---
        
        if "age" in data and data["age"] and "Age" in features_row.columns:
            features_row["Age"] = pd.to_numeric(data["age"])
        if "sex" in data and data["sex"] and "Sex" in features_row.columns:
            features_row["Sex"] = data["sex"] 
        if district not in df_gnn.index:
            return jsonify({"error": f"District '{district}' not found in GNN data."}), 404
        
        gnn_features = df_gnn.loc[[district]]
        
        # --- FIX: Merge on index ---
        final_features = pd.merge(features_row, gnn_features, left_index=True, right_index=True)
        # --- END OF FIX ---

        final_input_df = final_features[EXPECTED_COLUMNS]
        processed_data = preprocessor.transform(final_input_df)
        scaled_data = scaler.transform(processed_data)
        prediction_encoded = model.predict(scaled_data)
        prediction_class = label_encoder.inverse_transform(prediction_encoded)[0]
        return jsonify({
            "predicted_class": prediction_class,
            "district": district
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500


# --- 5. RUN THE SERVER ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)
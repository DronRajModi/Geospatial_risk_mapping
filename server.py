import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
import traceback
import sys 


try:
    from src.pipeline.predict_pipeline import PredictPipeline, CustomData
    from src.exception import CustomException
except ImportError as ie:
    print(f"ERROR: Failed to import pipeline modules. Make sure 'src' is in your PYTHONPATH. {ie}")
    sys.exit(1)

app = Flask(__name__)
CORS(app) 
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

#  2. LOAD ALL ARTIFACTS
try:
    artifacts_dir = "artifacts"
    model = joblib.load(os.path.join(artifacts_dir, "model_final_spatial_rf.pkl"))
    preprocessor = joblib.load(os.path.join(artifacts_dir, "final_preprocessor.pkl"))
    scaler = joblib.load(os.path.join(artifacts_dir, "final_scaler.pkl"))
    label_encoder = joblib.load(os.path.join(artifacts_dir, "label_encoder.pkl"))

    df_gnn = pd.read_csv(os.path.join(artifacts_dir, "gnn_embeddings.csv"))
    df_gnn['District'] = df_gnn['District'].str.strip().str.title()
    df_gnn = df_gnn.set_index('District')
    
  
    df_base_data = pd.read_csv(os.path.join(artifacts_dir, "data.csv"))
    df_base_data['District'] = df_base_data['District'].str.strip().str.title()
    df_district_templates = df_base_data.drop_duplicates(subset=['District']).set_index('District')
   

    feature_sets = [cols for name, trans, cols in preprocessor.transformers_]
    EXPECTED_COLUMNS = [col for subset in feature_sets for col in subset]

    # Create a synthetic population to test against
    SYNTHETIC_POPULATION_TEMPLATE = []
    ages = [25, 35, 45, 55, 65]
    sexes = ["Male", "Female"]
    for age in ages:
        for sex in sexes:
            SYNTHETIC_POPULATION_TEMPLATE.append({"Age": age, "Sex": sex})
    
    print(f" Server is ready. Loaded {len(df_district_templates)} district templates.")
    print(f" Synthetic population size: {len(SYNTHETIC_POPULATION_TEMPLATE)}")

except FileNotFoundError as e:
    print(f" ERROR: Missing artifact file. Server cannot start. {e}")
    model = None
except Exception as e:
    print(f" ERROR loading artifacts: {e}")
    traceback.print_exc()
    model = None


#  HEATMAP ENDPOINT 
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


# PREDICT ENDPOINT 
@app.route("/predict", methods=["POST"])
def predict():
    try:
     
        pipeline = PredictPipeline()
        
        
        data = request.json
        
        # We must cast numerical values correctly from the JSON
        custom_data = CustomData(
            District=data.get('District'),
            Age=int(data.get('Age')),
            Gender=data.get('Gender'),
            Tobacco_Use=data.get('Tobacco_Use'),
            Alcohol_Use=data.get('Alcohol_Use'),
            Hypertension=data.get('Hypertension'),
            Diabetes=data.get('Diabetes'),
            Obese=float(data.get('Obese')),
            Cholesterol=float(data.get('Cholesterol')),
            Sleep_Hours=float(data.get('Sleep_Hours')),
            Urban_or_Rural=data.get('Urban_or_Rural')
        )
        
        print(f"Received personal risk prediction request for: {custom_data.District}")
        
   
        prediction, confidence, pred_probs = pipeline.predict(custom_data)
        
        # 3b. Age risk profile
        age_labels, age_scores = pipeline.get_age_risk_profile(custom_data, prediction)
        
        # 3c. GNN Neighbors (for the "Analyze Neighbors" button)
        neighbors = pipeline.get_signature_neighbors(custom_data.District)
        
        # 3d. Feature importances
        importances = pipeline.get_feature_importances()
        
        # 3e. Lifestyle "what-if"
        lifestyle_what_if = pipeline.get_lifestyle_what_if(custom_data, pred_probs)

        # 4. Bundle everything into one JSON response for the frontend
        return jsonify({
            "success": True,
            "main_prediction": {
                "disease": prediction,
                "confidence": confidence
            },
            "age_risk_profile": {
                "labels": age_labels,
                "scores": age_scores
            },
            "spatial_neighbors": neighbors,
            "top_risk_factors": importances,
            "lifestyle_tips": lifestyle_what_if
        })
        
    except CustomException as e:
        # Handle exceptions from your pipeline
        print(f"Prediction Error (Custom): {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        # Handle general errors (e.g., missing key from JSON, type conversion error)
        print(f"Prediction Error (General): {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Server Error: {e}"}), 500

@app.route("/get_neighbors", methods=["GET"])
def get_neighbors():
    district_name = request.args.get("district")
    if not district_name:
        return jsonify({"error": "No district specified"}), 400
    
    try:
        # Instantiate pipeline just to get access to the GNN data
        pipeline = PredictPipeline() 
        neighbors = pipeline.get_signature_neighbors(district_name)
        return jsonify({"success": True, "neighbors": neighbors})

    except CustomException as e:
        print(f"Neighbor Error (Custom): {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        print(f"Neighbor Error (General): {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Server Error: {e}"}), 500

# --- 5. RUN THE SERVER ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)
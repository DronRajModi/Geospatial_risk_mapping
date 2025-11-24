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


@app.route("/get_risk_heatmap", methods=["GET"])
def get_risk_heatmap():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    target_disease = request.args.get("disease") 
    if not target_disease:
        return jsonify({"error": "No disease specified"}), 400
    
    # Force target disease to UPPERCASE to match React
    target_disease_upper = target_disease.upper()
    
    if target_disease not in label_encoder.classes_ and target_disease_upper not in label_encoder.classes_:
         return jsonify({"error": f"Disease '{target_disease}' not recognized."}), 400

    print(f"Generating heatmap for: {target_disease}")
    
    try:
        risk_data = {}
        
        imputed_districts = set(df_district_templates.index.unique())
        gnn_districts = set(df_gnn.index.unique())
        all_districts = list(imputed_districts.intersection(gnn_districts))
        print(f"Found {len(all_districts)} common districts to process.")

        for district in all_districts:
            try:
                features_row_template = df_district_templates.loc[[district]]
                gnn_features = df_gnn.loc[[district]]                
                full_features_template = pd.merge(features_row_template, gnn_features, left_index=True, right_index=True)

             
                pop_df_list = []
                for person_template in SYNTHETIC_POPULATION_TEMPLATE:
                    person_row = full_features_template.copy()
                 
                    person_row["Age"] = person_template["Age"]
                    person_row["Sex"] = person_template["Sex"]
                    pop_df_list.append(person_row)
                
                final_pop_df = pd.concat(pop_df_list)
                final_input_df = final_pop_df[EXPECTED_COLUMNS]
                
               
                processed_data = preprocessor.transform(final_input_df)
                scaled_data = scaler.transform(processed_data)
                
                
                predictions_proba = model.predict_proba(scaled_data)
                
                target_disease_index = np.where(label_encoder.classes_ == target_disease)[0][0]
                avg_disease_risk = predictions_proba[:, target_disease_index].mean()
                
                risk_percentage = round(avg_disease_risk * 100, 2)

                district_key = district.split(' (')[0].strip().upper()
                risk_data[district_key] = risk_percentage

            except KeyError as ke:
                continue 
            except Exception as e:
                continue

        print(f"Heatmap generation complete. Populated data for {len(risk_data)} districts.")
        return jsonify(risk_data)

    except Exception as e:
        print(f"Heatmap Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
     
        pipeline = PredictPipeline()
        
        
        data = request.json
        
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
        
       
        age_labels, age_scores = pipeline.get_age_risk_profile(custom_data, prediction)
        
      
        neighbors = pipeline.get_signature_neighbors(custom_data.District)
        
       
        importances = pipeline.get_feature_importances()
        
        
        lifestyle_what_if = pipeline.get_lifestyle_what_if(custom_data, pred_probs)

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
       
        print(f"Prediction Error (Custom): {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        
        print(f"Prediction Error (General): {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Server Error: {e}"}), 500

@app.route("/get_neighbors", methods=["GET"])
def get_neighbors():
    district_name = request.args.get("district")
    if not district_name:
        return jsonify({"error": "No district specified"}), 400
    
    try:
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
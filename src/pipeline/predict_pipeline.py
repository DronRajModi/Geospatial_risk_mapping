
import os
import sys
import pickle
import pandas as pd
import numpy as np
import argparse
from src.exception import CustomException
from src.logger import logger as logging
from src.utils import load_all_artifacts 
from sklearn.metrics.pairwise import cosine_similarity

class PredictPipeline:
    def __init__(self):
        try:
        
           
            (
                self.model, 
                self.preprocessor, 
                self.scaler, 
                self.label_encoder, 
                self.regional_data_lookup, 
                self.fallback_data,
                self.df_gnn_indexed 
            ) = load_all_artifacts()
         
            
            # Get the final feature order from the preprocessor
            self.num_cols = self.preprocessor.named_transformers_['num'].feature_names_in_
            self.cat_cols = self.preprocessor.named_transformers_['cat'].feature_names_in_
            self.gnn_cols = self.preprocessor.named_transformers_['gnn'].feature_names_in_
            self.all_feature_names = list(self.num_cols) + list(self.cat_cols) + list(self.gnn_cols)

        except Exception as e:
            raise CustomException(e, sys)

    def _load_regional_data(self):
       
        pass

    def _get_regional_data(self, district_name):
        """Fetches a single row of regional data for a given district."""
        try:
            regional_data = self.regional_data_lookup.loc[district_name]
            logging.info(f"Found regional data for district: {district_name}")
        except KeyError:
            logging.warning(f"District '{district_name}' not found. Using country-wide average as fallback.")
            regional_data = self.fallback_data
        return regional_data

    def _prepare_input_df(self, custom_data):
        """Combines user data and regional data into a single DataFrame row."""
        user_data_dict = custom_data.get_data_as_dict()
        district_name = str(user_data_dict.get('District')).strip().title()
        regional_data = self._get_regional_data(district_name)
        
        final_data_dict = user_data_dict
        for col in self.all_feature_names:
            if col not in final_data_dict:
                final_data_dict[col] = regional_data.get(col)
        
        return pd.DataFrame(final_data_dict, index=[0])[self.all_feature_names]

    def _run_prediction(self, custom_data):
        """Helper function to run the full predict pipeline and return arrays."""
        input_df = self._prepare_input_df(custom_data)
        logging.info("Applying preprocessor and scaler...")
        X_processed = self.preprocessor.transform(input_df)
        X_scaled = self.scaler.transform(X_processed)
        return X_scaled

    def predict(self, custom_data):
        """Runs the main prediction and returns class name and confidence."""
        try:
            logging.info("Starting main prediction...")
            X_scaled = self._run_prediction(custom_data)
            
            logging.info("Making prediction...")
            pred_probs = self.model.predict_proba(X_scaled)
            
            prediction_numeric = np.argmax(pred_probs, axis=1)[0]
            confidence_score = pred_probs[0][prediction_numeric]
            prediction_named = self.label_encoder.inverse_transform([prediction_numeric])[0]
            
            logging.info(f"Prediction: {prediction_named}, Confidence: {confidence_score:.2f}")
            return prediction_named, f"{confidence_score*100:.0f}%", pred_probs
        except Exception as e:
            raise CustomException(e, sys)

    def get_age_risk_profile(self, custom_data, predicted_disease):
        """Runs 'what-if' analysis to get risk scores for a disease across ages."""
        try:
            logging.info(f"Generating age-risk profile for: {predicted_disease}")
            target_class_index = np.where(self.label_encoder.classes_ == predicted_disease)[0][0]
            
            age_labels = list(range(20, 81, 5))
            age_scores = []

            for age in age_labels:
                custom_data.Age = age 
                X_scaled = self._run_prediction(custom_data)
                pred_probs = self.model.predict_proba(X_scaled)
                risk_score = pred_probs[0][target_class_index]
                age_scores.append(round(risk_score * 100, 2))
            
            logging.info("Age-risk profile complete.")
            return age_labels, age_scores
        except Exception as e:
            raise CustomException(e, sys)

    def get_signature_neighbors(self, district_name):
        """
        Uses GNN embeddings to find the most "spatially similar" districts.
        """
        try:
            logging.info(f"Finding GNN signature neighbors for: {district_name}")
            target_district = str(district_name).strip().title()
            
           
            target_vector = self.df_gnn_indexed.loc[[target_district]]
            similarity_matrix = cosine_similarity(target_vector, self.df_gnn_indexed)
            similarity_series = pd.Series(similarity_matrix[0], index=self.df_gnn_indexed.index)
            
            
            top_neighbors = similarity_series.sort_values(ascending=False)[1:6]
            logging.info(f"Found neighbors: {top_neighbors.index.tolist()}")
            return top_neighbors.index.tolist()

        except KeyError:
            logging.warning(f"District '{district_name}' not in GNN data. Cannot find neighbors.")
            return ["N/A (District not in GNN database)"]
        except Exception as e:
            raise CustomException(e, sys)

    def get_feature_importances(self):
        """
        Gets the top feature importances from the RandomForest model.
        """
        try:
            logging.info("Calculating feature importances...")
            
            importances = self.model.feature_importances_
            
         
            cat_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out()
            
            all_final_features = list(self.num_cols) + list(cat_feature_names) + list(self.gnn_cols)
            df_imp = pd.DataFrame({'Feature': all_final_features, 'Importance': importances})

            original_feature_map = {}
            for col in self.num_cols: original_feature_map[col] = col
            for col in self.gnn_cols: original_feature_map[col] = "GNN Spatial Signature"
            for col in cat_feature_names:
                original_name = col.split('_')[0]
                original_feature_map[col] = original_name

            df_imp['Original_Feature'] = df_imp['Feature'].map(original_feature_map)
            df_grouped_imp = df_imp.groupby('Original_Feature')['Importance'].sum().sort_values(ascending=False)
            
            top_5_features = df_grouped_imp.head(5)
            return top_5_features.to_dict()

        except Exception as e:
            raise CustomException(e, sys)
            
    def get_lifestyle_what_if(self, custom_data, original_probs):
        """
        Runs 'what-if' analysis on controllable lifestyle factors.
        """
        try:
            logging.info("Running lifestyle 'what-if' analysis...")
            results = {}
            original_pred_class = np.argmax(original_probs)
            original_pred_prob = original_probs[0][original_pred_class]

            # 1. What if user stops using tobacco?
            if custom_data.Tobacco_Use == "Yes":
                custom_data_no_tobacco = CustomData(**custom_data.get_data_as_dict(as_kwargs=True))
                custom_data_no_tobacco.Tobacco_Use = "No"
                X_scaled = self._run_prediction(custom_data_no_tobacco)
                new_probs = self.model.predict_proba(X_scaled)[0]
                change = (new_probs[original_pred_class] - original_pred_prob) * 100
                results['Stop Tobacco Use'] = f"Risk changes by {change:+.0f}%"

            # 2. What if user stops using alcohol?
            if custom_data.Alcohol_Use == "Yes":
                custom_data_no_alcohol = CustomData(**custom_data.get_data_as_dict(as_kwargs=True))
                custom_data_no_alcohol.Alcohol_Use = "No"
                X_scaled = self._run_prediction(custom_data_no_alcohol)
                new_probs = self.model.predict_proba(X_scaled)[0]
                change = (new_probs[original_pred_class] - original_pred_prob) * 100
                results['Stop Alcohol Use'] = f"Risk changes by {change:+.0f}%"
                
            # 3. What if user sleeps 8 hours?
            if custom_data.Sleep_Hours < 7.5:
                custom_data_more_sleep = CustomData(**custom_data.get_data_as_dict(as_kwargs=True))
                custom_data_more_sleep.Sleep_Hours = 8.0
                X_scaled = self._run_prediction(custom_data_more_sleep)
                new_probs = self.model.predict_proba(X_scaled)[0]
                change = (new_probs[original_pred_class] - original_pred_prob) * 100
                results['Sleep 8 Hours'] = f"Risk changes by {change:+.0f}%"
            
            if not results:
                return {"Info": "Your lifestyle factors are already optimal!"}
            
            return results
            
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 District: str, Age: int, Gender: str,
                 Tobacco_Use: str, Alcohol_Use: str, Hypertension: str,
                 Diabetes: str, Obese: float, Cholesterol: float,
                 Sleep_Hours: float, Urban_or_Rural: str):
        
        self.District = District
        self.Age = Age
        self.Gender = Gender
        self.Tobacco_Use = Tobacco_Use
        self.Alcohol_Use = Alcohol_Use
        self.Hypertension = Hypertension
        self.Diabetes = Diabetes
        self.Obese = Obese
        self.Cholesterol = Cholesterol
        self.Sleep_Hours = Sleep_Hours
        self.Urban_or_Rural = Urban_or_Rural
    
    def get_data_as_dict(self, as_kwargs=False):
        try:
            data_dict = {
                "District": self.District, "Age": self.Age, "Gender": self.Gender,
                "Tobacco_Use": self.Tobacco_Use, "Alcohol_Use": self.Alcohol_Use,
                "Hypertension": self.Hypertension, "Diabetes": self.Diabetes,
                "Obese": self.Obese, "Cholesterol": self.Cholesterol,
                "Sleep_Hours": self.Sleep_Hours, "Urban_or_Rural": self.Urban_or_Rural
            }
            if as_kwargs:
                return data_dict

            # Map to CSV column names
            data_dict["Obese(%)"] = data_dict.pop("Obese")
            data_dict["Cholesterol(%)"] = data_dict.pop("Cholesterol")
            return data_dict
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    logging.info("Running prediction pipeline from command line...")
    
    parser = argparse.ArgumentParser(description='Predict NCD risk.')
    parser.add_argument('--District', type=str, default='New Delhi')
    parser.add_argument('--Age', type=int, default=55)
    parser.add_argument('--Gender', type=str, default='Male')
    parser.add_argument('--Tobacco_Use', type=str, default='Yes')
    parser.add_argument('--Alcohol_Use', type=str, default='Yes')
    parser.add_argument('--Hypertension', type=str, default='Yes')
    parser.add_argument('--Diabetes', type=str, default='No')
    parser.add_argument('--Obese', type=float, default=28.5)
    parser.add_argument('--Cholesterol', type=float, default=45.0)
    parser.add_argument('--Sleep_Hours', type=float, default=6.0)
    parser.add_argument('--Urban_or_Rural', type=str, default='Urban')
    
    args = parser.parse_args()

    data = CustomData(
        District=args.District, Age=args.Age, Gender=args.Gender,
        Tobacco_Use=args.Tobacco_Use, Alcohol_Use=args.Alcohol_Use,
        Hypertension=args.Hypertension, Diabetes=args.Diabetes,
        Obese=args.Obese, Cholesterol=args.Cholesterol,
        Sleep_Hours=args.Sleep_Hours, Urban_or_Rural=args.Urban_or_Rural
    )
    
    pipeline = PredictPipeline()
    
    # Test 1: Main Prediction
    prediction, confidence, pred_probs = pipeline.predict(data)
    print("\n--- Prediction ---")
    print(f"Predicted Disease: {prediction}")
    print(f"Confidence: {confidence}")
    
    # Test 2: Age Risk Plot
    age_labels, age_scores = pipeline.get_age_risk_profile(data, prediction)
    print("\n--- Age Risk Profile ---")
    print(f"Ages: {age_labels}")
    print(f"Risk Scores (%): {age_scores}")
    
    # Test 3: GNN Neighbors
    neighbors = pipeline.get_signature_neighbors(data.District)
    print("\n--- GNN Signature Neighbors ---")
    print(f"Districts spatially similar to {data.District}: {neighbors}")

    # Test 4: Feature Importance
    importances = pipeline.get_feature_importances()
    print("\n--- Top 5 Model Risk Factors ---")
    if importances:
        for feature, importance in importances.items():
            print(f"{feature}: {importance*100:.2f}%")
        
    # Test 5: Lifestyle What-If
    lifestyle_results = pipeline.get_lifestyle_what_if(data, pred_probs)
    print("\n--- Lifestyle 'What-If' Analysis ---")
    if lifestyle_results:
        for scenario, result in lifestyle_results.items():
            print(f"If you '{scenario}': {result}")
    else:
        print("Your lifestyle factors are already optimal!")
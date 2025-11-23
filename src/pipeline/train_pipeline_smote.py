
import os
import sys
from src.exception import CustomException
from src.logger import logger as logging


from src.components.spatial_imputation import SpatialImputer

from src.components.data_transformation_smote import initiate_data_transformation_smote
from src.components.model_trainer import initiate_model_training


class TrainPipelineSMOTE:
    def __init__(self):
      
        self.artifacts_dir = "artifacts"
        self.train_data_path = os.path.join(self.artifacts_dir, "train.csv")
        self.test_data_path = os.path.join(self.artifacts_dir, "test.csv")
        self.raw_data_path = os.path.join(self.artifacts_dir, "data.csv")

    def run_pipeline(self):
        logging.info("Main SMOTE-NC training pipeline started")
        try:
        
            logging.info("STEP 1: Data Ingestion (Skipped, using existing artifacts)")
            
            # Stage A - Spatial Imputation 
            logging.info("STEP 2: Starting Stage A: Spatial Imputation")
            SHAPEFILE_DISTRICT_COL = 'NAME_2' 
            spatial_imputer_obj = SpatialImputer()
            imputed_features_path = spatial_imputer_obj.initiate_spatial_imputation(
                district_col_name='District',
                shapefile_district_col=SHAPEFILE_DISTRICT_COL 
            )
            logging.info(f"Stage A complete. Full imputed feature set at {imputed_features_path}")

            
            # Data Transformation & Augmentation (Stage B with SMOTE-NC) 
            logging.info("STEP 3: Running Data Transformation (with SMOTE-NC augmentation)")
          
            train_arr_scaled_aug, test_arr_scaled, preprocessor_path, scaler_path = initiate_data_transformation_smote(
                train_input=self.train_data_path,
                test_input=self.test_data_path,
                artifacts_dir=self.artifacts_dir
            )
            logging.info("Data Transformation and Augmentation complete.")

            
            #  Model Trainer (Original RandomForest) 
            logging.info("STEP 4: Running Model Trainer")
            model_trainer_obj = initiate_model_training(
                train_arr=train_arr_scaled_aug, 
                test_arr=test_arr_scaled,   
                artifacts_dir=self.artifacts_dir
            )
            logging.info(f"Model Trainer complete. Accuracy: {model_trainer_obj['accuracy']:.4f}")

            logging.info("--- Stages C (GNN) and D (LSTM) will be added here later ---")

            
            logging.info("Main SMOTE-NC training pipeline finished successfully")

        except Exception as e:
            logging.error(f"An error occurred in the main SMOTE-NC pipeline: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise CustomException(e, sys)



if __name__ == "__main__":
    pipeline = TrainPipelineSMOTE()
    pipeline.run_pipeline()
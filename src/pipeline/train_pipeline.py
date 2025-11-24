import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.spatial_imputation import SpatialImputer


class TrainPipeline:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir

    def run_pipeline(self):
        logging.info("Main training pipeline started")
        try:
            logging.info("Running Data Ingestion")
            ingest_obj = DataIngestion(artifacts_dir=self.artifacts_dir)
            ingestion_result = ingest_obj.initiate_data_ingestion()
            logging.info(f"Data Ingestion complete. Training + test data processed.")

            train_data_path = f"{self.artifacts_dir}/train.csv"
            test_data_path = f"{self.artifacts_dir}/test.csv"

            logging.info("Starting Stage A: Spatial Imputation")
            SHAPEFILE_DISTRICT_COL = 'NAME_2'  
            spatial_imputer_obj = SpatialImputer()
            imputed_features_path = spatial_imputer_obj.initiate_spatial_imputation(
                district_col_name='District', 
                shapefile_district_col=SHAPEFILE_DISTRICT_COL
            )
            logging.info(f"Stage A complete. Imputed features saved at {imputed_features_path}")

            logging.info("Running Data Transformation")
            data_transform_obj = DataTransformation()
            train_arr, test_arr, transformer_obj_path = data_transform_obj.initiate_data_transformation(
                train_data_path,
                test_data_path,
                artifacts_dir=self.artifacts_dir
            )
            logging.info(f"Data Transformation complete. train_arr shape: {train_arr.shape}, test_arr shape: {test_arr.shape}")
            logging.info("Running Model Trainer")
            model_trainer_obj = ModelTrainer()
            training_result = model_trainer_obj.initiate_model_training(
                train_arr,
                test_arr,
                artifacts_dir=self.artifacts_dir
            )
            logging.info(f"Model training complete. Model saved at {training_result['model_path']}")
            logging.info(f"Accuracy: {training_result['accuracy']:.4f}, F1 (weighted): {training_result['f1_weighted']:.4f}")

            logging.info("Main training pipeline finished successfully")

        except Exception as e:
            logging.error("An error occurred in the training pipeline")
            raise CustomException(e, sys)



if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()

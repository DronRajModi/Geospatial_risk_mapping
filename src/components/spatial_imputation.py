import os
import sys
import pandas as pd
import geopandas as gpd
from sklearn.impute import KNNImputer
from dataclasses import dataclass

# Import your custom logger and exception
from src.exception import CustomException
from src.logger import logging

@dataclass
class SpatialImputationConfig:
    # Path to the new file we will create
    imputed_data_path: str = os.path.join('artifacts', 'imputed_features.csv')
    # Input shapefile (full map)
    shapefile_path: str = os.path.join('notebook', 'india_districts.shp')
    # Input data file (partial data)
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class SpatialImputer:
    def __init__(self):
        self.imputation_config = SpatialImputationConfig()

    def initiate_spatial_imputation(self, district_col_name='District', shapefile_district_col='DISTRICT'):
        """
        Main function to perform spatial imputation.
        
        Args:
            district_col_name (str): The column name for 'District' in your data.csv.
            shapefile_district_col (str): The column name for 'District' in your .shp file.
                                            (You must confirm this name!)
        """
        logging.info("Starting Spatial Imputation (Stage A)")
        
        try:
            # 1. Load Data
            logging.info(f"Loading partial data from {self.imputation_config.raw_data_path}")
            df = pd.read_csv(self.imputation_config.raw_data_path)

            # 2. Aggregate Data: Create one average profile per district
            # We must exclude non-numeric columns from the mean calculation
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            # Remove 'Year' if it's not a feature you want to average
            if 'Year' in numeric_cols:
                numeric_cols.remove('Year')
            
            # Group by district and get the mean of all numeric features
            df_agg = df.groupby(district_col_name)[numeric_cols].mean().reset_index()
            logging.info(f"Aggregated {len(df_agg)} districts to their mean profiles.")

            # 3. Load Map
            logging.info(f"Loading full map from {self.imputation_config.shapefile_path}")
            gdf = gpd.read_file(self.imputation_config.shapefile_path)
            
            # Ensure district names match for merging
            # You might need to adjust this logic if names are different
            gdf[district_col_name] = gdf[shapefile_district_col].str.strip().str.title()
            
            # 4. Merge
            logging.info(f"Merging full map ({len(gdf)} districts) with partial data ({len(df_agg)} districts)")
            gdf_merged = gdf.merge(df_agg, on=district_col_name, how='left')
            
            # Keep only the essential columns: district, geometry, and features
            features_to_impute = [district_col_name, 'geometry'] + numeric_cols
            gdf_impute = gdf_merged[features_to_impute]

            # 5. Impute
            logging.info("Preparing for KNN Imputation")
            # Create centroid coordinates to use for distance calculation
            gdf_impute['centroid_x'] = gdf_impute.geometry.centroid.x
            gdf_impute['centroid_y'] = gdf_impute.geometry.centroid.y
            
            # These are the columns the imputer will use
            impute_cols = ['centroid_x', 'centroid_y'] + numeric_cols
            
            # Create a copy to avoid SettingWithCopyWarning
            imputer_data = gdf_impute[impute_cols].copy()
            
            # Run the KNN Imputer
            logging.info("Running KNNImputer (k=5) based on spatial centroids...")
            imputer = KNNImputer(n_neighbors=5, weights="distance")
            imputed_array = imputer.fit_transform(imputer_data)
            
            # Convert back to DataFrame
            df_imputed = pd.DataFrame(imputed_array, columns=impute_cols)
            
            # 6. Save
            # Combine the imputed data with the original district/geometry info
            final_df = pd.concat(
                [gdf_impute[[district_col_name, 'geometry']].reset_index(drop=True),
                 df_imputed.drop(['centroid_x', 'centroid_y'], axis=1).reset_index(drop=True)],
                axis=1
            )
            
            os.makedirs(os.path.dirname(self.imputation_config.imputed_data_path), exist_ok=True)
            # We save as a CSV. We lose the geometry, but Stage C (GNN) can re-add it.
            # Or we can save as a GeoPackage (gpkg)
            final_df_csv = final_df.drop('geometry', axis=1)
            final_df_csv.to_csv(self.imputation_config.imputed_data_path, index=False)
            
            logging.info(f"Spatial Imputation complete. Full dataset saved to {self.imputation_config.imputed_data_path}")
            
            return self.imputation_config.imputed_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # This is to test the component by itself
    
    # !!! IMPORTANT: Check your shapefile and find the real column name for districts !!!
    # Then replace 'DISTRICT' with the correct name.
    SHAPEFILE_DISTRICT_COLUMN = 'NAME_2' # <-- CHECK THIS
    
    obj = SpatialImputer()
    imputed_path = obj.initiate_spatial_imputation(
        district_col_name='District',
        shapefile_district_col=SHAPEFILE_DISTRICT_COLUMN
    )
    print(f"Test run complete. Imputed data saved to: {imputed_path}")
    df = pd.read_csv(imputed_path)
    print("\nImputed Data Head:")
    print(df.head())
    print(f"\nTotal districts in new file: {len(df)}")
    print(f"Any NaN values left? {df.isnull().sum().sum()}")
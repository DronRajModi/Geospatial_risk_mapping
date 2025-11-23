import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logger as logging

@dataclass
class LSTMModelTrainerConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    gnn_embeddings_path: str = os.path.join('artifacts', 'gnn_embeddings.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    
    preprocessor_path: str = os.path.join('artifacts', 'transformer_smote.pkl')
    scaler_path: str = os.path.join('artifacts', 'scaler_smote.pkl')
    label_encoder_path: str = os.path.join('artifacts', 'label_encoder.pkl')
    
    lstm_model_path: str = os.path.join('artifacts', 'lstm_model.h5')
    lookback: int = 3 

class LSTMModelTrainer:
    def __init__(self):
        self.config = LSTMModelTrainerConfig()
        try:
            with open(self.config.preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            with open(self.config.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(self.config.label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            logging.info("Loaded preprocessor, scaler, and label encoder from Stage B artifacts.")
        except Exception as e:
            raise CustomException(e, sys)

    def _load_and_merge_data(self):
        try:
            logging.info("Loading raw data and GNN embeddings...")
            df = pd.read_csv(self.config.raw_data_path)
            df_gnn = pd.read_csv(self.config.gnn_embeddings_path)
            
            df['District'] = df['District'].str.strip().str.title()
            df_gnn['District'] = df_gnn['District'].str.strip().str.title()
            
            df_merged = pd.merge(df, df_gnn, on='District', how='left')
            df_merged.dropna(subset=['Disease_NCD'], inplace=True) 
            
            gnn_cols = [col for col in df_merged.columns if col.startswith('gnn_embed_')]
            df_merged[gnn_cols] = df_merged[gnn_cols].fillna(0)
            
            logging.info(f"Data loaded and merged. Shape: {df_merged.shape}")
            return df_merged
            
        except Exception as e:
            raise CustomException(e, sys)

    def _create_sequences(self, df, district_list):
        try:
            logging.info("Creating sequences...")
            X_sequences, y_sequences = [], []
            
            cat_cols = self.preprocessor.transformers_[1][2]
            num_cols = self.preprocessor.transformers_[0][2]
            gnn_cols = [col for col in df.columns if col.startswith('gnn_embed_')]
            
            X_df_features = df.drop(columns=['Disease_NCD', 'Year'])
            
            X_processed = self.preprocessor.transform(X_df_features)
            X_scaled = self.scaler.transform(X_processed)
            
            X_gnn = df[gnn_cols].values
            X_final_features = np.hstack((X_scaled, X_gnn))
            
            df_final_features = pd.DataFrame(X_final_features)
            df_final_features['District'] = df['District'].values
            df_final_features['Year'] = df['Year'].values
            df_final_features['Target'] = self.label_encoder.transform(df['Disease_NCD'])
            
            for district in district_list:
                df_dist = df_final_features[df_final_features['District'] == district].sort_values(by='Year')
                
                features = df_dist.drop(columns=['District', 'Year', 'Target']).values
                targets = df_dist['Target'].values
                
                if len(features) < self.config.lookback + 1:
                    continue 

                for i in range(len(features) - self.config.lookback):
                    X_sequences.append(features[i : i + self.config.lookback])
                    y_sequences.append(targets[i + self.config.lookback]) 
            
            return np.array(X_sequences), np.array(y_sequences)

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_lstm_training(self):
        try:
            df_merged = self._load_and_merge_data()
            
            train_districts = pd.read_csv(self.config.train_data_path)['District'].str.strip().str.title().unique()
            test_districts = pd.read_csv(self.config.test_data_path)['District'].str.strip().str.title().unique()
            
            X_train_seq, y_train_seq = self._create_sequences(df_merged, train_districts)
            X_test_seq, y_test_seq = self._create_sequences(df_merged, test_districts)
            
            if X_train_seq.shape[0] == 0:
                raise ValueError("No training sequences were created. Check data and lookback period.")
            
            logging.info(f"Training sequences created: X={X_train_seq.shape}, y={y_train_seq.shape}")
            logging.info(f"Test sequences created: X={X_test_seq.shape}, y={y_test_seq.shape}")
            
            logging.info("Applying SMOTE to training sequences...")
            n_samples, n_timesteps, n_features = X_train_seq.shape
            X_train_2d = X_train_seq.reshape((n_samples, n_timesteps * n_features))
            
            min_class_count = pd.Series(y_train_seq).value_counts().min()
            k_neighbors_smote = max(1, min_class_count - 1)
            
            if k_neighbors_smote < 1:
                 logging.warning(f"Smallest class in sequences has {min_class_count} samples. Cannot apply SMOTE.")
                 X_train_aug_2d = X_train_2d
                 y_train_aug = y_train_seq
            else:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
                X_train_aug_2d, y_train_aug = smote.fit_resample(X_train_2d, y_train_seq)
            
            X_train_aug_3d = X_train_aug_2d.reshape((X_train_aug_2d.shape[0], n_timesteps, n_features))
            
            logging.info(f"SMOTE complete. New training shape: {X_train_aug_3d.shape}")

            logging.info("Building LSTM model...")
            n_classes = len(self.label_encoder.classes_)
            
            model = Sequential()
            model.add(LSTM(units=128, return_sequences=True, input_shape=(n_timesteps, n_features)))
            model.add(Dropout(0.3))
            model.add(BatchNormalization())
            model.add(LSTM(units=64))
            model.add(Dropout(0.3))
            model.add(BatchNormalization())
            model.add(Dense(units=32, activation='relu'))
            model.add(Dense(units=n_classes, activation='softmax')) 
            
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.summary()
            
            logging.info("Training LSTM model...")
            model.fit(X_train_aug_3d, y_train_aug, epochs=50, batch_size=64, validation_split=0.2, verbose=1)
            
            logging.info("Evaluating LSTM model on test set...")
            y_pred_probs = model.predict(X_test_seq)
            y_pred = np.argmax(y_pred_probs, axis=1) 
            
            report = classification_report(y_test_seq, y_pred, target_names=self.label_encoder.classes_, zero_division=0)
            
            logging.info(f"[LSTM Model] Classification Report:\n{report}")
            
            model.save(self.config.lstm_model_path)
            logging.info(f"LSTM model saved to {self.config.lstm_model_path}")
            
            return report

        except Exception as e:
            raise CustomException(e, sys)


import os
import sys
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from datetime import datetime

LOG_FILE = f"GAN_Experiment_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    format="[ %(asctime)s ] %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GANLogger")

def _load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

class DataAugmenterGAN:
    def __init__(self, feature_dim, latent_dim=32, num_classes=1):
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.cgan = self._build_cgan()

    def _build_generator(self):
        noise_input = layers.Input(shape=(self.latent_dim,))
        label_input = layers.Input(shape=(self.num_classes,))
        
        label_embedding = layers.Dense(self.latent_dim)(label_input)
        merged = layers.Concatenate()([noise_input, label_embedding])
        
        x = layers.Dense(64, activation='relu')(merged)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        output = layers.Dense(self.feature_dim, activation='linear')(x)
        
        return models.Model([noise_input, label_input], output)

    def _build_discriminator(self):
        feature_input = layers.Input(shape=(self.feature_dim,))
        label_input = layers.Input(shape=(self.num_classes,))
        
        label_embedding = layers.Dense(self.feature_dim)(label_input)
        merged = layers.Concatenate()([feature_input, label_embedding])
        
        x = layers.Dense(128, activation='relu')(merged)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model([feature_input, label_input], output)

    def _build_cgan(self):
        optimizer = optimizers.Adam(0.0002, 0.5)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.discriminator.trainable = False
        
        noise = layers.Input(shape=(self.latent_dim,))
        label = layers.Input(shape=(self.num_classes,))
        img = self.generator([noise, label])
        valid = self.discriminator([img, label])
        
        model = models.Model([noise, label], valid)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        return model

    def train(self, X_train, y_train, target_class, epochs=2000, batch_size=32):
        logger.info(f"Training GAN for class {target_class}...")
        
        # Filter data for the specific class
        X_class = X_train[y_train == target_class]
        
        if len(X_class) < 5:
            logger.warning(f"Not enough samples ({len(X_class)}) to train GAN for class {target_class}. Skipping.")
            return False

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # Train Discriminator
            idx = np.random.randint(0, X_class.shape[0], batch_size)
            real_features = X_class[idx]
            labels = np.full((batch_size, 1), target_class)
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_features = self.generator.predict([noise, labels], verbose=0)
            
            d_loss_real = self.discriminator.train_on_batch([real_features, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_features, labels], fake)
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.cgan.train_on_batch([noise, labels], valid)
            
            if epoch % 500 == 0:
                logger.info(f"{epoch}/{epochs} [D loss: {d_loss_real[0]}] [G loss: {g_loss}]")
        
        return True

    def generate(self, target_class, n_samples):
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        labels = np.full((n_samples, 1), target_class)
        gen_features = self.generator.predict([noise, labels], verbose=0)
        return gen_features

def run_gan_pipeline():
    try:
        logger.info("STEP 1: Loading Data...")
        train_path = os.path.join('artifacts', 'train.csv')
        test_path = os.path.join('artifacts', 'test.csv')
        
        df_train = pd.read_csv(train_path).dropna(subset=['Disease_NCD'])
        df_test = pd.read_csv(test_path).dropna(subset=['Disease_NCD'])
        
        target = 'Disease_NCD'
        X_train = df_train.drop(columns=[target, 'District', 'Year'], errors='ignore')
        y_train = df_train[target]
        X_test = df_test.drop(columns=[target, 'District', 'Year'], errors='ignore')
        y_test = df_test[target]

        logger.info("STEP 2: Preprocessing Data...")
        
        num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X_train.select_dtypes(include='object').columns.tolist()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
                ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
            ])
        
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)
        
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        
        logger.info("STEP 3: Training GANs for Minority Classes...")
        
        class_counts = pd.Series(y_train_enc).value_counts()
        minority_classes = class_counts[class_counts < 500].index.tolist() # Threshold of 500
        
        X_train_final = X_train_scaled
        y_train_final = y_train_enc
        
        gan = DataAugmenterGAN(feature_dim=X_train_scaled.shape[1], num_classes=1)
        
        for cls in minority_classes:
            cls_name = le.inverse_transform([cls])[0]
            logger.info(f"--> Attempting GAN training for: {cls_name} (Original count: {class_counts[cls]})")
            
            success = gan.train(X_train_scaled, y_train_enc, target_class=cls, epochs=2000)
            
            if success:
                logger.info(f"    Generating 1000 synthetic samples for {cls_name}...")
                X_fake = gan.generate(target_class=cls, n_samples=1000)
                y_fake = np.full(1000, cls)
                
                X_train_final = np.vstack([X_train_final, X_fake])
                y_train_final = np.hstack([y_train_final, y_fake])
        
        logger.info(f"Augmentation Complete. New Training Size: {X_train_final.shape[0]}")

        logger.info("STEP 4: Training Random Forest on GAN-Augmented Data...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_final, y_train_final)
        
        logger.info("STEP 5: Evaluating GAN+RF Model...")
        preds = rf.predict(X_test_scaled)
        
        acc = accuracy_score(y_test_enc, preds)
        report = classification_report(y_test_enc, preds, target_names=le.classes_, zero_division=0)
        
        logger.info(f"GAN+RF Accuracy: {acc:.4f}")
        logger.info(f"GAN+RF Classification Report:\n{report}")
        
        print("\n--- GAN + Random Forest Results ---")
        print(f"Accuracy: {acc:.4f}")
        print(report)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_gan_pipeline()
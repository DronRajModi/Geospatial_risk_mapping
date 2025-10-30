# src/components/data_augmentation.py

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataAugmentationConfig:
    # Optional: Path to save trained generator model
    generator_model_path: str = os.path.join('artifacts', 'cgan_generator.h5')

class DataAugmenter:
    """
    A Conditional GAN (cGAN) to augment tabular data.
    It learns to generate samples for a specific class.
    """
    def __init__(self, feature_dim, latent_dim=32, num_classes=1):
        self.config = DataAugmentationConfig()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        # num_classes=1 because we pass the class label (0, 1, 2, etc.) as a single number
        self.num_classes = num_classes 

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.cgan = self._build_cgan()

    def _build_generator(self):
        # Input 1: Random noise
        noise_input = layers.Input(shape=(self.latent_dim,))
        # Input 2: Class label (e.g., 0, 1, 2, 3, 4)
        label_input = layers.Input(shape=(self.num_classes,))
        
        # Embed the label to match noise dimensions for better conditioning
        label_embedding = layers.Dense(self.latent_dim)(label_input)
        
        # Concatenate noise and embedded label
        merged_input = layers.Concatenate()([noise_input, label_embedding])

        model = layers.Dense(64, activation='relu')(merged_input)
        model = layers.BatchNormalization()(model)
        model = layers.Dense(128, activation='relu')(model)
        model = layers.BatchNormalization()(model)
        # Output layer has 'feature_dim' neurons to match the input data
        output = layers.Dense(self.feature_dim, activation='linear')(model) # 'linear' as data is scaled

        return models.Model([noise_input, label_input], output, name="Generator")

    def _build_discriminator(self):
        # Input 1: Features (real or fake)
        feature_input = layers.Input(shape=(self.feature_dim,))
        # Input 2: Class label
        label_input = layers.Input(shape=(self.num_classes,))

        # Embed the label
        label_embedding = layers.Dense(self.feature_dim)(label_input)
        
        # Concatenate features and embedded label
        merged_input = layers.Concatenate()([feature_input, label_embedding])

        model = layers.Dense(128, activation='relu')(merged_input)
        model = layers.Dropout(0.3)(model)
        model = layers.Dense(64, activation='relu')(model)
        model = layers.Dropout(0.3)(model)
        # Output is a single probability (real or fake)
        output = layers.Dense(1, activation='sigmoid')(model)

        return models.Model([feature_input, label_input], output, name="Discriminator")

    def _build_cgan(self):
        # --- Optimizers & Loss ---
        generator_optimizer = optimizers.Adam(0.0002, 0.5)
        discriminator_optimizer = optimizers.Adam(0.0002, 0.5)
        binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()

        # --- Compile Discriminator ---
        self.discriminator.compile(loss=binary_cross_entropy, optimizer=discriminator_optimizer, metrics=['accuracy'])
        
        # --- Build Combined Model (to train Generator) ---
        self.discriminator.trainable = False # Freeze discriminator weights

        noise_in = layers.Input(shape=(self.latent_dim,))
        label_in = layers.Input(shape=(self.num_classes,))
        
        generated_features = self.generator([noise_in, label_in])
        validity = self.discriminator([generated_features, label_in])

        cgan_model = models.Model([noise_in, label_in], validity, name="cGAN")
        cgan_model.compile(loss=binary_cross_entropy, optimizer=generator_optimizer)
        return cgan_model

    def train_gan(self, X_train_scaled, y_train, target_class, epochs=10000, batch_size=32, log_interval=1000):
        logging.info(f"Starting cGAN Training for class {target_class}...")
        try:
            # We will train the GAN ONLY on samples from the target minority class
            minority_features_scaled = X_train_scaled[y_train == target_class]
            
            if len(minority_features_scaled) == 0:
                 logging.warning(f"No samples found for class {target_class}. Skipping GAN training.")
                 return False # Indicate failure
            
            # Adjust batch size if minority class is smaller than batch size
            current_batch_size = min(batch_size, len(minority_features_scaled))
            if current_batch_size < batch_size:
                logging.warning(f"Minority class size ({len(minority_features_scaled)}) is less than batch size ({batch_size}). Using batch size {current_batch_size}.")


            # Labels for conditioning
            real_labels = np.full((current_batch_size, self.num_classes), target_class)
            # Labels for discriminator training
            real_y = np.ones((current_batch_size, 1)) * 0.9 # Label smoothing
            fake_y = np.zeros((current_batch_size, 1))
            generator_y = np.ones((current_batch_size, 1)) # Labels for training generator

            for epoch in range(epochs):
                # --- Train Discriminator ---
                # Select a random batch of real minority samples
                idx = np.random.randint(0, minority_features_scaled.shape[0], current_batch_size)
                real_features_batch = minority_features_scaled[idx]
                
                # Generate a batch of fake samples for the same minority class
                noise = np.random.normal(0, 1, (current_batch_size, self.latent_dim))
                generated_features_batch = self.generator.predict([noise, real_labels], verbose=0)

                # Train discriminator
                d_loss_real = self.discriminator.train_on_batch([real_features_batch, real_labels], real_y)
                d_loss_fake = self.discriminator.train_on_batch([generated_features_batch, real_labels], fake_y)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # --- Train Generator ---
                noise = np.random.normal(0, 1, (current_batch_size, self.latent_dim))
                g_loss = self.cgan.train_on_batch([noise, real_labels], generator_y)

                if (epoch + 1) % log_interval == 0:
                    logging.info(f"GAN Epoch {epoch+1}/{epochs} (Class {target_class}) | D Loss: {d_loss[0]:.4f} (Acc: {d_loss[1]*100:.2f}%) | G Loss: {g_loss:.4f}")

            logging.info(f"cGAN Training Finished for class {target_class}.")
            return True # Indicate success

        except Exception as e:
            raise CustomException(e, sys)

    def generate_samples(self, target_class, n_samples):
        logging.info(f"Generating {n_samples} synthetic samples for class {target_class}...")
        try:
            noise_gen = np.random.normal(0, 1, (n_samples, self.latent_dim))
            synthetic_labels = np.full((n_samples, self.num_classes), target_class)
            
            synthetic_features_scaled = self.generator.predict([noise_gen, synthetic_labels])
            synthetic_labels_array = np.full(n_samples, target_class)
            
            return synthetic_features_scaled, synthetic_labels_array

        except Exception as e:
            raise CustomException(e, sys)

# --- Main function to be called from the pipeline ---
def initiate_data_augmentation(train_arr_scaled, test_arr_scaled, rows_to_generate=1000, gan_epochs=10000, gan_batch_size=32):
    """
    Takes scaled train/test arrays (last column is target), performs GAN augmentation
    on the training set, and returns augmented scaled train array and
    original scaled test array.
    """
    try:
        logging.info("Initiating Data Augmentation Component (Stage B)")
        
        X_train_scaled = train_arr_scaled[:, :-1]
        y_train = train_arr_scaled[:, -1]
        
        feature_dim = X_train_scaled.shape[1]
        
        # --- Identify Minority Classes to Augment ---
        class_counts = pd.Series(y_train).value_counts().sort_values()
        logging.info(f"Original training class distribution:\n{class_counts.to_string()}")
        
        # Find all classes that are NOT the majority class
        majority_class_count = class_counts.max()
        minority_classes_df = class_counts[class_counts < majority_class_count]
        
        X_train_scaled_aug = X_train_scaled
        y_train_aug = y_train
        
        augmenter = DataAugmenter(feature_dim=feature_dim)
        
        # --- Generate 1000 rows FOR EACH minority class ---
        # Based on your report, classes 0 and 4 are the ones with 0.00 precision.
        
        # We will augment ANY class that is not the majority
        classes_to_augment = minority_classes_df.index.tolist()
        
        if not classes_to_augment:
            logging.info("Dataset is already balanced. Skipping augmentation.")
            return train_arr_scaled, test_arr_scaled

        logging.info(f"Will attempt to augment minority classes: {classes_to_augment}")

        for class_label in classes_to_augment:
            class_label_int = int(class_label)
            logging.info(f"--- Processing augmentation for Class {class_label_int} ---")
            
            # 1. Train a GAN specifically for this class
            success = augmenter.train_gan(
                X_train_scaled, 
                y_train, 
                target_class=class_label_int, 
                epochs=gan_epochs, 
                batch_size=gan_batch_size
            )
            
            if success:
                # 2. Generate the 1000 new samples
                synthetic_features, synthetic_labels = augmenter.generate_samples(
                    target_class=class_label_int, 
                    n_samples=rows_to_generate
                )
                
                # 3. Add them to the augmented dataset
                X_train_scaled_aug = np.vstack((X_train_scaled_aug, synthetic_features))
                y_train_aug = np.concatenate((y_train_aug, synthetic_labels))
            else:
                logging.warning(f"GAN training failed or skipped for class {class_label_int}.")


        # --- Final Shuffle ---
        logging.info("Shuffling augmented dataset...")
        shuffle_indices = np.random.permutation(len(X_train_scaled_aug))
        X_train_scaled_aug = X_train_scaled_aug[shuffle_indices]
        y_train_aug = y_train_aug[shuffle_indices]

        logging.info(f"Augmented training data shape: {X_train_scaled_aug.shape}")
        logging.info(f"New class distribution:\n{pd.Series(y_train_aug).value_counts().to_string()}")

        # Recombine features and labels into augmented train array
        train_arr_scaled_aug = np.hstack((X_train_scaled_aug, y_train_aug.reshape(-1, 1)))

        logging.info("Data Augmentation Component Finished")
        # Return augmented scaled train array and ORIGINAL scaled test array
        return train_arr_scaled_aug, test_arr_scaled

    except Exception as e:
        raise CustomException(e, sys)
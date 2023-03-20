import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedStratifiedKFold,
)

from tensorflow import keras
from keras import layers, losses
from keras.models import Model

import random, time, sys

from models import *


class AnomalyDetector:
    """Implementation of anomaly detector using AEs and GANs"""

    def __init__(self):
        """Initializes the anomaly detector, parses arguments"""

        # e.g. sys.argv = [main.py AE pandex/trace/ProcessAll.csv pandex/trace/trace_pandex_merged.csv]
        if len(sys.argv) < 4:
            print("ERROR: Not enough arguments.")
            sys.exit(1)

        if sys.argv[1].lower() not in ["ae", "aae", "adae"]:
            print("ERROR: Invalid model type.")
            sys.exit(1)

        self.path = "../data/"
        self.data_path, self.label_path = sys.argv[2], sys.argv[3]
        self.model_type = sys.argv[1].lower()

        try:
            open(self.path + self.data_path)
        except Exception as err:
            print("ERROR: Processes file")
            print(err)
            sys.exit(1)

        try:
            open(self.path + self.label_path)
        except Exception as err:
            print("ERROR: Labels file")
            print(err)
            sys.exit(1)

        self.model_type_dict = {
            "ae": "AutoEncoder",
            "aae": "Adversarial AutoEncoder",
            "adae": "Adversarial Dual AutoEncoder",
        }

        self.cross_entropy = losses.BinaryCrossentropy()

    def load_data(self):
        """Loads data based on command line arguments"""
        processes = pd.read_csv(self.path + self.data_path)
        labels_df = pd.read_csv(self.path + self.label_path)
        apt_list = labels_df.loc[labels_df["label"] == "AdmSubject::Node"]["uuid"]

        if "Object_ID" in processes.columns:
            col = "Object_ID"
        else:
            col = "UUID"

        labels_series = processes[col].isin(apt_list)
        self.labels = labels_series.values
        self.data = processes.values[:, 1:]
        print("Load data: finished.")
        print(f"Data dimension: {self.data.shape}")

    def prepare_data(self, test_size=0.2):
        """Prepares loaded data

        Args:
            test_size: proportion of dataset to reserve for test set
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=42
        )

        self.X_train = tf.cast(self.X_train.astype(np.int32), tf.int32)
        self.X_test = tf.cast(self.X_test.astype(np.int32), tf.int32)
        self.normal_X_train = self.X_train[~self.y_train]
        self.normal_X_test = self.X_test[~self.y_test]
        self.normal_X = tf.concat([self.normal_X_train, self.normal_X_test], 0)
        self.anomalous_X_train = self.X_train[self.y_train]
        self.anomalous_X_test = self.X_test[self.y_test]
        self.anomalous_X = tf.concat([self.anomalous_X_train, self.anomalous_X_test], 0)
        print("Prepare data: finished.")
        print(f"Normal data points: {self.normal_X.shape[0]}")
        print(f"Anomalous data points: {self.anomalous_X.shape[0]}")

    def create_models(self):
        """Creates anomaly detection models"""
        print(f"Selected model: {self.model_type_dict[self.model_type]}")

        # Hard coding hidden layers architecture
        # TODO: update this to take in command line input
        self.hidden_dims = [128, 64, 32, 16, 8]

        self.AE = AutoEncoder(
            hidden_dims=self.hidden_dims, output_shape=self.normal_X.shape[1]
        )
        self.AAE = AdversarialAutoEncoder(
            hidden_dims=self.hidden_dims, output_shape=self.normal_X.shape[1]
        )
        self.ADAE = AdversarialDualAutoEncoder(
            hidden_dims=self.hidden_dims, output_shape=self.normal_X.shape[1]
        )

    # Case: AutoEncoder

    def ae_loss(self, input_data, reconstructed_output):
        return self.cross_entropy(input_data, reconstructed_output)

    # Case: Adversarial AutoEncoder

    def aae_generator_loss(self, input_data, reconstructed_output):
        return self.cross_entropy(input_data, reconstructed_output)

    def aae_discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def aae_combined_loss(
        self,
        input_data,
        reconstructed_output,
        real_output,
        fake_output,
        lambda_value=0.5,
    ):
        gen_loss = self.aae_generator_loss(input_data, reconstructed_output)
        disc_loss = self.aae_discriminator_loss(real_output, fake_output)
        return gen_loss + lambda_value * disc_loss

    # Case: Adversarial Dual AutoEncoder

    def adae_generator_loss(self, input_data, gen_output, disc_output):
        return self.cross_entropy(input_data, gen_output) + self.cross_entropy(
            gen_output, disc_output
        )

    def adae_discriminator_loss(self, input_data, gen_output, real_output, fake_output):
        real_loss = self.cross_entropy(input_data, real_output)
        fake_loss = self.cross_entropy(gen_output, fake_output)
        return real_loss + fake_loss

    @tf.function
    def train_step_ae(self, x, optimizer):
        with tf.GradientTape() as tape:
            reconstructed_data = self.AE(x, training=True)
            loss = self.ae_loss(x, reconstructed_data)
        grads = tape.gradient(loss, self.AE.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.AE.trainable_variables))
        return loss

    def train_model_ae(self, lr=0.002, epochs=20, batch_size=512):
        """Training function for AutoEncoder (AE) model"""
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_samples = self.normal_X.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.losses_mean = []
        print("\nTraining starting")
        for epoch in range(self.epochs):
            start = time.time()
            print(f"\nTraining at epoch {epoch+1}, ", end="")
            losses = []
            for batch_index in range(self.num_batches):
                x = self.normal_X[
                    batch_index * self.batch_size : (batch_index + 1) * self.batch_size
                ]
                if x.shape[0] == self.batch_size:
                    loss = self.train_step_ae(x, self.optimizer)
                    losses.append(loss)
            print("time = %.5f sec." % (time.time() - start))
            print("\tMean loss = %.10f" % (np.mean(losses)))
            self.losses_mean.append(np.mean(losses))

    @tf.function
    def train_step_aae(self, x, gen_optimizer, disc_optimizer):
        noise_dim = self.normal_X.shape[1]
        noise = tf.random.normal([self.batch_size, noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.AAE.generator(noise, training=True)

            real_output = self.AAE.discriminator(x, training=True)
            fake_output = self.AAE.discriminator(generated_data, training=True)

            gen_loss = self.aae_combined_loss(
                x, generated_data, real_output, fake_output, lambda_value=0.2
            )
            disc_loss = self.aae_discriminator_loss(real_output, fake_output)

        gen_grads = gen_tape.gradient(
            gen_loss, self.AAE.generator.trainable_variables
        )
        disc_grads = disc_tape.gradient(
            disc_loss, self.AAE.discriminator.trainable_variables
        )

        gen_optimizer.apply_gradients(
            zip(gen_grads, self.AAE.generator.trainable_variables)
        )
        disc_optimizer.apply_gradients(
            zip(disc_grads, self.AAE.discriminator.trainable_variables)
        )

        return gen_loss, disc_loss

    def train_model_aae(self, lr=0.002, epochs=20, batch_size=512):
        """Training function for Adversarial AutoEncoder (AAE)"""
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_samples = self.normal_X.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

        self.gen_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.disc_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.gen_losses_mean, self.disc_losses_mean = [], []
        print("\nTraining starting")
        for epoch in range(self.epochs):
            start = time.time()
            print(f"\nTraining at epoch {epoch+1}, ", end="")
            gen_losses, disc_losses = [], []
            for batch_index in range(self.num_batches):
                x = self.normal_X[
                    batch_index * self.batch_size : (batch_index + 1) * self.batch_size
                ]
                if x.shape[0] == self.batch_size:
                    gen_loss, disc_loss = self.train_step_aae(x, self.gen_optimizer, self.disc_optimizer)
                    gen_losses.append(gen_loss)
                    disc_losses.append(disc_loss)
            print("time = %.5f sec." % (time.time() - start))
            print(
                "\tMean gen_loss = %.10f; mean disc_loss = %.10f"
                % (np.mean(gen_losses), np.mean(disc_losses))
            )
            self.gen_losses_mean.append(np.mean(gen_losses))
            self.disc_losses_mean.append(np.mean(disc_losses))

    @tf.function
    def train_step_adae(self, x, gen_optimizer, disc_optimizer):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.ADAE.generator(x, training=True)

            real_output = self.ADAE.discriminator(x, training=True)
            fake_output = self.ADAE.discriminator(generated_data, training=True)

            gen_loss = self.adae_generator_loss(x, generated_data, fake_output)
            disc_loss = self.adae_discriminator_loss(x, generated_data, real_output, fake_output)

        gen_grads = gen_tape.gradient(gen_loss, self.ADAE.generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.ADAE.discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(gen_grads, self.ADAE.generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(disc_grads, self.ADAE.discriminator.trainable_variables))

        return gen_loss, disc_loss
    
    def train_model_adae(self, lr=0.002, epochs=20, batch_size=512):
        """Training function for Adversarial Dual AutoEncoder (ADAE)"""
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_samples = self.normal_X.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

        self.gen_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.disc_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.gen_losses_mean, self.disc_losses_mean = [], []
        print("\fTraining starting")
        for epoch in range(epochs):
            start = time.time()
            print(f"\nTraining at epoch {epoch+1}, ", end="")
            gen_losses, disc_losses = [], []
            for batch_index in range(self.num_batches):
                x = self.normal_X[
                    batch_index * self.batch_size : (batch_index + 1) * self.batch_size
                ]
                if x.shape[0] == self.batch_size:
                    gen_loss, disc_loss = self.train_step_adae(x, self.gen_optimizer, self.disc_optimizer)
                    gen_losses.append(gen_loss)
                    disc_losses.append(disc_loss)
            print("time = %.5f sec." % (time.time() - start))
            print(
                "\tMean gen_loss = %.10f; mean disc_loss = %.10f"
                % (np.mean(gen_losses), np.mean(disc_losses))
            )
            self.gen_losses_mean.append(np.mean(gen_losses))
            self.disc_losses_mean.append(np.mean(disc_losses))

def main():
    AD = AnomalyDetector()
    AD.load_data()
    AD.prepare_data()
    AD.create_models()
    if AD.model_type == "ae":
        AD.train_model_ae()
        print("\nFinished training AE. Losses mean:")
        print(AD.losses_mean)
    elif AD.model_type == "aae":
        AD.train_model_aae()
        print("\nFinished training AAE.")
        print("Gen losses mean:")
        print(AD.gen_losses_mean)
        print("Disc losses mean:")
        print(AD.disc_losses_mean)
    elif AD.model_type == "adae":
        AD.train_model_adae()
        print("\nFinished training ADAE.")
        print("Gen losses mean:")
        print(AD.gen_losses_mean)
        print("Disc losses mean:")
        print(AD.disc_losses_mean)

if __name__ == "__main__":
    main()

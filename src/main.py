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

        self.load_data()
        self.prepare_data()

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
        """Prepared loaded data

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

    def define_losses(self):
        """Defines loss functions based on model type"""
        cross_entropy = losses.BinaryCrossentropy()
        # Case: AutoEncoder
        if self.model_type == "ad":

            def ae_loss(self, input_data, reconstructed_output):
                return cross_entropy(input_data, reconstructed_output)

        # Case: Adversarial AutoEncoder
        elif self.model_type == "aae":

            def generator_loss(self, input_data, reconstructed_output):
                return cross_entropy(input_data, reconstructed_output)

            def discriminator_loss(self, real_output, fake_output):
                real_loss = cross_entropy(tf.ones_like(real_output), real_output)
                fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
                return real_loss + fake_loss

            def combined_loss(
                self,
                input_data,
                reconstructed_output,
                real_output,
                fake_output,
                lambda_value=0.5,
            ):
                gen_loss = generator_loss(input_data, reconstructed_output)
                disc_loss = discriminator_loss(real_output, fake_output)
                return gen_loss + lambda_value * disc_loss

        # Case: Adversarial Dual AutoEncoder
        else:

            def generator_loss(self, input_data, gen_output, disc_output):
                return cross_entropy(input_data, gen_output) + cross_entropy(
                    gen_output, disc_output
                )

            def discriminator_loss(input_data, gen_output, real_output, fake_output):
                real_loss = cross_entropy(input_data, real_output)
                fake_loss = cross_entropy(gen_output, fake_output)
                return real_loss + fake_loss

    def train_model(self, lr=0.002):
        """TODO"""
        optimizer = keras.optimizers.legacy.Adam(learning_rate=lr)


def main():
    AD = AnomalyDetector()
    print("Hello, world!")


if __name__ == "__main__":
    main()

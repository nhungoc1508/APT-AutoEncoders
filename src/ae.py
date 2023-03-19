import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold

from tensorflow import keras
from keras import layers, losses
from keras.models import Model

import random, time

class AutoEncoder(Model):
    """Implementation of AutoEncoder

    Attrs:
        hidden_dims: list including numbers of units in hidden layers of encoder and decoder
        output_shape: dimension of output = dimension of input data
    """
    def __init__(self, hidden_dims, output_shape):
        """Initializes the model with a defined architecture

        Args:
            hidden_dims: defines the symmetric architecture of encoder and decoder
                        e.g. hidden_dim = [32, 16, 8] -- encoder has 3 layers with 32, 16, 8 units,
                        decoder has 3 layers with 16, 32, output_shape units
            output_shape: defines shape of output
        """
        super(AutoEncoder, self).__init__()
        self.hidden_dims = hidden_dims

        # Create encoder
        self.encoder_layers = []
        for dim in self.hidden_dims:
            self.encoder_layers.append(layers.Dense(units=dim, activation='relu'))
        self.encoder = keras.Sequential(self.encoder_layers)

        # Create decoder
        self.decoder_layers = []
        for dim in self.hidden_dims[len(self.hidden_dims)-2:0:-1]:
            self.decoder_layers.append(layers.Dense(units=dim, activation='relu'))
        self.decoder_layers.append(layers.Dense(unit=output_shape, activation='sigmoid'))
        self.decoder = keras.Sequential(self.decoder_layers)

    def call(self, x):
        """Defines the call function

        Args:
            x: input data
        
        Returns:
            reconstructed data
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
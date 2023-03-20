import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model


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
            self.encoder_layers.append(layers.Dense(units=dim, activation="relu"))
        self.encoder = keras.Sequential(self.encoder_layers)

        # Create decoder
        self.decoder_layers = []
        for dim in self.hidden_dims[len(self.hidden_dims) - 2 : 0 : -1]:
            self.decoder_layers.append(layers.Dense(units=dim, activation="relu"))
        self.decoder_layers.append(
            layers.Dense(units=output_shape, activation="sigmoid")
        )
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


class AdversarialAutoEncoder(Model):
    """Implementation of AdversarialAutoEncoder

    Attrs:
        hidden_dims: list including numbers of units in hidden layers of encoder and decoder
        output_shape: dimension of output = dimension of input data
    """

    def __init__(self, hidden_dims, output_shape):
        """Initializes the model with a defined architecture

        Args:
            hidden_dims: defines the symmetric architecture of encoder and decoder
                        e.g. hidden_dim = [32, 16, 8] -- encoder has 3 layers with 32, 16, 8 units,
                        decoder has 3 layers with 16, 32, output_shape units,
                        discriminator has 3 layers with 32, 16, 1 units
            output_shape: defines shape of output
        """
        super(AdversarialAutoEncoder, self).__init__()
        self.hidden_dims = hidden_dims

        # Create generator
        self.generator = AutoEncoder(
            hidden_dims=self.hidden_dims, output_shape=output_shape
        )

        # Create discriminator
        self.discriminator_layers = []
        for dim in self.hidden_dims[: len(self.hidden_dims) - 1]:
            self.discriminator_layers.append(layers.Dense(units=dim, activation="relu"))
        self.discriminator_layers.append(layers.Dense(units=1, activation="sigmoid"))
        self.discriminator = keras.Sequential(self.discriminator_layers)

    def call(self, x):
        """Defines the call function

        Args:
            x: input data

        Returns:
            discriminator's prediction on whether data is real or generated by generator
        """
        generated = self.generator(x)
        discriminated = self.discriminator(generated)
        return discriminated


class AdversarialDualAutoEncoder(Model):
    """Implementation of AdversarialDualAutoEncoder

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
        super(AdversarialDualAutoEncoder, self).__init__()
        self.hidden_dims = hidden_dims

        self.generator = AutoEncoder(
            hidden_dims=self.hidden_dims, output_shape=output_shape
        )
        self.discriminator = AutoEncoder(
            hidden_dims=self.hidden_dims, output_shape=output_shape
        )

    def call(self, x):
        """Defines the call function

        Args:
            x: input data

        Returns:
            reconstruced input data (by discriminator)
        """
        generated = self.generator(x)
        discriminated = self.discriminator(generated)
        return discriminated
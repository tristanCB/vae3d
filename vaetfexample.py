# Taken from https://www.tensorflow.org/guide/keras/custom_layers_and_models
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

# Force use CPU.
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Fetch MNIST
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("uint8") / 255

## reshape to be linear
# mnist_digits = mnist_digits.reshape((mnist_digits.shape[0],
#                                     mnist_digits.shape[1]*mnist_digits.shape[2]))
original_dim = (mnist_digits.shape[1], mnist_digits.shape[2], 1)
intermediate_dim = 16
latent_dim = 3

# Define encoder model.
original_inputs = tf.keras.Input(shape=original_dim, name="encoder_input")
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(original_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(intermediate_dim, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(inputs=original_inputs, outputs=[z_mean,z_log_var,z], name="encoder")

# Define decoder model.
latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")

# x = layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

class VAE(tf.keras.Model):
    """
    https://keras.io/examples/generative/vae/
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def call(self, inputs):
        return 

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

# Load Models
try:
    encoder = tf.keras.models.load_model("ENCODER")
    decoder = tf.keras.models.load_model("DECODER")
except:
    print("Could not load model, pass to training phase")
    pass

vae = VAE(encoder, decoder)
vae.compile(tf.keras.optimizers.Adam())


class CustomCallback(tf.keras.callbacks.Callback):
    """
    https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
    """
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))
        # Save Models
        encoder.save("ENCODER")
        decoder.save("DECODER")

# 
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.2, patience=3, min_lr=0.00001
)

# Train
vae.fit(mnist_digits, mnist_digits, epochs=200, batch_size=128, callbacks=[CustomCallback(),reduce_lr])

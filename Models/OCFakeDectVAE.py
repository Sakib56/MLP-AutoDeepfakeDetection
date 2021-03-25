from __future__ import division
import tensorflow as tf
import keras
from keras import layers
from keras import metrics

LATENT_DIM = 20_000
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 100, 100, 3

# Build VAE models as custom (keras model class) class
# OCFakeDect1 = encoder+sampling+decoder
class OCFakeDect1(keras.Model):
    def __init__(self, **kwargs):
        super(OCFakeDect1, self).__init__(**kwargs)
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        
    def create_encoder(self):
        # OC-FakeDect Encoder
        encoder_inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        # 100x100x3
        x = layers.Conv2D(16, 3, strides=1, padding="same")(encoder_inputs)
        x = layers.BatchNormalization(axis = 1)(x)
        x = layers.Activation('relu')(x)
        # 100x100x16
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization(axis = 1)(x)
        x = layers.Activation('relu')(x)
        # 50x50x32
        x = layers.Conv2D(64, 3, strides=1, padding="same")(x)
        x = layers.BatchNormalization(axis = 1)(x)
        x = layers.Activation('relu')(x)
        # 50x50x64
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization(axis = 1)(x)
        x = layers.Activation('relu')(x)
        # 25x25x32
        x = layers.Flatten()(x)
        # 20000
        # Sampling
        z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
        z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        # Defining full encoder as keras model
        return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        
    def create_decoder(self):
        # OC-FakeDect Decoder
        latent_inputs = keras.Input(shape=(LATENT_DIM,))
        # 20000
        x = layers.Reshape((25, 25, 32))(latent_inputs)
        x = layers.Conv2DTranspose(32, 3, strides=1, padding="same")(x)
        x = layers.BatchNormalization(axis = 1)(x)
        x = layers.Activation('relu')(x)
        # 25x25x32
        x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization(axis = 1)(x)
        x = layers.Activation('relu')(x)
        # 50x50x64
        x = layers.Conv2DTranspose(32, 3, strides=1, padding="same")(x)
        x = layers.BatchNormalization(axis = 1)(x)
        x = layers.Activation('relu')(x)
        # 50x50x32
        x = layers.Conv2DTranspose(16, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization(axis = 1)(x)
        x = layers.Activation('relu')(x)
        # 100x100x16
        x = layers.Conv2DTranspose(3, 3, strides=1, padding="same")(x)
        x = layers.BatchNormalization(axis = 1)(x)
        decoder_outputs = layers.Activation('sigmoid')(x)
        # 100x100x3
        # Defining full decoder as keras model
        return keras.Model(latent_inputs, decoder_outputs, name="decoder")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(inputs, reconstruction)
        )
        reconstruction_loss *= 28 * 28
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        self.add_metric(total_loss, name='total_loss', aggregation='mean')
        self.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
        return reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(data, reconstruction)
        )
        reconstruction_loss *= 28 * 28
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        return {"loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss}

# OCFakeDect1 = encoder+sampling+decoder+encoder+sampling
# class OCFakeDect2(keras.Model):
#     def __init__(self, **kwargs):
#         super(OCFakeDect2, self).__init__(**kwargs)
#         self.encoder = self.create_encoder()
#         self.decoder = self.create_decoder()
#         self.total_loss_tracker = metrics.Mean(name="total_loss")
#         self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
#         self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        
#     def create_encoder(self):
#         # OC-FakeDect Encoder
#         encoder_inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#         # 100x100x3
#         x = layers.Conv2D(16, 3, strides=1, padding="same")(encoder_inputs)
#         x = layers.BatchNormalization(axis = 1)(x)
#         x = layers.Activation('relu')(x)
#         # 100x100x16
#         x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
#         x = layers.BatchNormalization(axis = 1)(x)
#         x = layers.Activation('relu')(x)
#         # 50x50x32
#         x = layers.Conv2D(64, 3, strides=1, padding="same")(x)
#         x = layers.BatchNormalization(axis = 1)(x)
#         x = layers.Activation('relu')(x)
#         # 50x50x64
#         x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
#         x = layers.BatchNormalization(axis = 1)(x)
#         x = layers.Activation('relu')(x)
#         # 25x25x32
#         x = layers.Flatten()(x)
#         # 20000
#         # Sampling
#         z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
#         z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
#         z = Sampling()([z_mean, z_log_var])
#         # Defining full encoder as keras model
#         return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        
#     def create_decoder(self):
#         # OC-FakeDect Decoder
#         latent_inputs = keras.Input(shape=(LATENT_DIM,))
#         # 20000
#         x = layers.Reshape((25, 25, 32))(latent_inputs)
#         x = layers.Conv2DTranspose(32, 3, strides=1, padding="same")(x)
#         x = layers.BatchNormalization(axis = 1)(x)
#         x = layers.Activation('relu')(x)
#         # 25x25x32
#         x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
#         x = layers.BatchNormalization(axis = 1)(x)
#         x = layers.Activation('relu')(x)
#         # 50x50x64
#         x = layers.Conv2DTranspose(32, 3, strides=1, padding="same")(x)
#         x = layers.BatchNormalization(axis = 1)(x)
#         x = layers.Activation('relu')(x)
#         # 50x50x32
#         x = layers.Conv2DTranspose(16, 3, strides=2, padding="same")(x)
#         x = layers.BatchNormalization(axis = 1)(x)
#         x = layers.Activation('relu')(x)
#         # 100x100x16
#         x = layers.Conv2DTranspose(3, 3, strides=1, padding="same")(x)
#         x = layers.BatchNormalization(axis = 1)(x)
#         x = layers.Activation('sigmoid')(x)
#         # 100x100x3
#         x = layers.Conv2D(16, 3, strides=1, padding="same")(x)
#         x = layers.BatchNormalization(axis = 1)(x)
#         x = layers.Activation('relu')(x)
#         # 100x100x16
#         x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
#         x = layers.BatchNormalization(axis = 1)(x)
#         x = layers.Activation('relu')(x)
#         # 50x50x32
#         x = layers.Conv2D(64, 3, strides=1, padding="same")(x)
#         x = layers.BatchNormalization(axis = 1)(x)
#         x = layers.Activation('relu')(x)
#         # 50x50x64
#         x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
#         x = layers.BatchNormalization(axis = 1)(x)
#         x = layers.Activation('relu')(x)
#         # 25x25x32
#         x = layers.Flatten()(x)
#         # 20000
#         # Sampling
#         z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
#         z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
#         z = Sampling()([z_mean, z_log_var])
#         return keras.Model(latent_inputs, [z_mean, z_log_var, z], name="decoder+encoder")

#     @property
#     def metrics(self):
#         return [self.total_loss_tracker,
#                 self.reconstruction_loss_tracker,
#                 self.kl_loss_tracker]

#     def call(self, inputs):
#         z_mean, z_log_var, z = self.encoder(inputs)
#         reconstruction = self.decoder(z)
#         reconstruction_loss = tf.reduce_mean(
#             keras.losses.binary_crossentropy(inputs, reconstruction)
#         )
#         reconstruction_loss *= 28 * 28
#         kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
#         kl_loss = tf.reduce_mean(kl_loss)
#         kl_loss *= -0.5
#         total_loss = reconstruction_loss + kl_loss
#         self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
#         self.add_metric(total_loss, name='total_loss', aggregation='mean')
#         self.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
#         return reconstruction

#     def train_step(self, data):
#         with tf.GradientTape() as tape:
#             z_mean, z_log_var, z = self.encoder(data)
#             reconstruction = self.decoder(z)
#             reconstruction_loss = tf.reduce_mean(
#                 tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))
#             kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
#             kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
#             total_loss = reconstruction_loss + kl_loss
#         grads = tape.gradient(total_loss, self.trainable_weights)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#         self.total_loss_tracker.update_state(total_loss)
#         self.reconstruction_loss_tracker.update_state(reconstruction_loss)
#         self.kl_loss_tracker.update_state(kl_loss)
#         return {"loss": self.total_loss_tracker.result(),
#                 "reconstruction_loss": self.reconstruction_loss_tracker.result(),
#                 "kl_loss": self.kl_loss_tracker.result()}

#     def test_step(self, data):
#         if isinstance(data, tuple):
#             data = data[0]
#         z_mean, z_log_var, z = self.encoder(data)
#         reconstruction = self.decoder(z)
#         reconstruction_loss = tf.reduce_mean(
#             keras.losses.binary_crossentropy(data, reconstruction)
#         )
#         reconstruction_loss *= 28 * 28
#         kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
#         kl_loss = tf.reduce_mean(kl_loss)
#         kl_loss *= -0.5
#         total_loss = reconstruction_loss + kl_loss
#         return {"loss": total_loss,
#                 "reconstruction_loss": reconstruction_loss,
#                 "kl_loss": kl_loss}

# Sampling Class, Random sampling function for latent vector, z; "variational part"
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
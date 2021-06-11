import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
from ast import literal_eval as make_tuple
from scipy.sparse import csr_matrix

# pretrain autoencoder
checkpoint_path = "autoencoder/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

from tensorflow.keras import layers


class VGGConvBlock(layers.Layer):
    def __init__(self, num_filters, block_id):
        super(VGGConvBlock, self).__init__(name="VGGConvBlock{}".format(block_id))
        self.conv1 = tfkl.Conv2D(filters=num_filters, kernel_size=(3, 3), activation='relu')
        self.conv2 = tfkl.Conv2D(filters=num_filters, kernel_size=(3, 3), activation='relu')
        self.maxpool = tfkl.MaxPooling2D((2, 2))

    def call(self, inputs, **kwargs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.maxpool(out)

        return out


class VGGDeConvBlock(layers.Layer):
    def __init__(self, num_filters, block_id):
        super(VGGDeConvBlock, self).__init__(name="VGGDeConvBlock{}".format(block_id))
        self.upsample = tfkl.UpSampling2D((2, 2), interpolation='bilinear')
        self.convT1 = tfkl.Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), padding='valid', activation='relu')
        self.convT2 = tfkl.Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), padding='valid', activation='relu')

    def call(self, inputs, **kwargs):
        out = self.upsample(inputs)
        out = self.convT1(out)
        out = self.convT2(out)

        return out


class VGGEncoder(layers.Layer):
    def __init__(self, encoded_size):
        super(VGGEncoder, self).__init__(name='VGGEncoder')
        self.layers = [VGGConvBlock(32, 1), VGGConvBlock(64, 2)]
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs, **kwargs):
        out = inputs

        # Iterate through blocks
        for block in self.layers:
            out = block(out)

        out_flat = tfkl.Flatten()(out)
        mu = self.mu(out_flat)
        sigma = self.sigma(out_flat)

        return mu, sigma


class VGGDecoder(layers.Layer):
    def __init__(self, input_shape, activation):
        super(VGGDecoder, self).__init__(name='VGGDecoder')

        input_tuple = make_tuple(input_shape)
        if input_tuple == (64, 64, 1):
            target_shape = (13, 13, 64)
        elif input_tuple == (64, 64, 3):
            target_shape = (13, 13, 64)
        elif input_tuple == (32, 32, 3):
            target_shape = (5, 5, 64)

        self.activation = activation
        self.dense = tfkl.Dense(target_shape[0] * target_shape[1] * target_shape[2])
        self.reshape = tfkl.Reshape(target_shape=target_shape)
        self.layers = [VGGDeConvBlock(64, 1), VGGDeConvBlock(32, 2)]
        self.convT = tfkl.Conv2DTranspose(filters=input_tuple[2], kernel_size=3, padding='same')

    def call(self, inputs, **kwargs):
        out = self.dense(inputs)
        out = self.reshape(out)

        # Iterate through blocks
        for block in self.layers:
            out = block(out)

        # Last convolution
        out = self.convT(out)

        if self.activation == "sigmoid":
            out = tf.sigmoid(out)

        return out


class CNNEncoder(layers.Layer):
    def __init__(self, encoded_size):
        super(CNNEncoder, self).__init__(name='CNNEncoder')
        self.conv1 = tfkl.Conv2D(filters=32, kernel_size=4, strides=(2, 2), activation='relu')
        self.conv2 = tfkl.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs, **kwargs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out_flat = tfkl.Flatten()(out)  # Should be 15x15x64 for heart_echo, 7x7x64 for cifar10
        mu = self.mu(out_flat)
        sigma = self.sigma(out_flat)

        return mu, sigma


class CNNDecoder(layers.Layer):
    def __init__(self, input_shape, activation):
        super(CNNDecoder, self).__init__(name='CNNDecoder')
        self.activation = activation

        # TODO: Make this better
        input_tuple = make_tuple(input_shape)
        if input_tuple == (64, 64, 1):
            target_shape = (15, 15, 64)
        elif input_tuple == (64, 64, 3):
            target_shape = (15, 15, 64)
        elif input_tuple == (32, 32, 3):
            target_shape = (7, 7, 64)

        # self.dense = tfkl.Dense(15 * 15 * 64, activation='relu')
        # self.reshape = tfkl.Reshape(target_shape=(15, 15, 64))
        self.dense = tfkl.Dense(target_shape[0] * target_shape[1] * target_shape[2])
        self.reshape = tfkl.Reshape(target_shape=target_shape)
        self.convT1 = tfkl.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='valid',
                                           activation='relu')
        self.convT2 = tfkl.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='valid',
                                           activation='relu')
        # self.convT3 = tfkl.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')
        self.convT3 = tfkl.Conv2DTranspose(filters=input_tuple[2], kernel_size=3, strides=1, padding='same')

    def call(self, inputs, **kwargs):
        out = self.dense(inputs)
        out = self.reshape(out)
        out = self.convT1(out)
        out = self.convT2(out)
        out = self.convT3(out)

        if self.activation == "sigmoid":
            out = tf.sigmoid(out)

        return out


class Encoder(layers.Layer):
    def __init__(self, encoded_size):
        super(Encoder, self).__init__(name='encoder')
        self.dense1 = tfkl.Dense(500, activation='relu')
        self.dense2 = tfkl.Dense(500, activation='relu')
        self.dense3 = tfkl.Dense(2000, activation='relu')
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs):
        x = tfkl.Flatten()(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma


class Decoder(layers.Layer):
    def __init__(self, input_shape, activation):
        super(Decoder, self).__init__(name='dec')
        self.inp_shape = input_shape
        self.dense1 = tfkl.Dense(2000, activation='relu')
        self.dense2 = tfkl.Dense(500, activation='relu')
        self.dense3 = tfkl.Dense(500, activation='relu')
        if activation == "sigmoid":
            print("yeah")
            self.dense4 = tfkl.Dense(self.inp_shape, activation="sigmoid")
        else:
            self.dense4 = tfkl.Dense(self.inp_shape)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x


class DCGMM(tf.keras.Model):
    def __init__(self, **kwargs):
        super(DCGMM, self).__init__(name="DCGMM")#, dynamic=True)
        self.encoded_size = kwargs['latent_dim']
        self.num_clusters = kwargs['num_clusters']
        self.inp_shape = kwargs['inp_shape']
        self.activation = kwargs['activation']
        self.type = kwargs['type']

        if self.type == "FC":
            self.encoder = Encoder(self.encoded_size)
            self.decoder = Decoder(self.inp_shape, self.activation)
        elif self.type == "CNN":
            self.encoder = CNNEncoder(self.encoded_size)
            self.decoder = CNNDecoder(self.inp_shape, self.activation)
        elif self.type == "VGG":
            self.encoder = VGGEncoder(self.encoded_size)
            self.decoder = VGGDecoder(self.inp_shape, self.activation)
        else:
            raise NotImplemented("Invalid type {}".format(self.type))

        self.c_mu = tf.Variable(tf.ones([self.num_clusters, self.encoded_size]), name="mu")
        self.log_c_sigma = tf.Variable(tf.ones([self.num_clusters, self.encoded_size]), name="sigma")
        self.prior = tf.constant(tf.ones([self.num_clusters]) * (
                1 / self.num_clusters))  # tf.Variable(tf.ones([self.num_clusters]), name="prior")

    def call(self, inputs, training=True):
        inputs, W = inputs
        z_mu, log_z_sigma = self.encoder(inputs)
        z = tfd.MultivariateNormalDiag(loc=z_mu, scale_diag=tf.math.sqrt(tf.math.exp(log_z_sigma)))
        z_sample = z.sample()

        log_z_sigma_tile = tf.expand_dims(log_z_sigma, axis=-2)
        c = tf.constant([1, self.num_clusters, 1], tf.int32)
        log_z_sigma_tile = tf.tile(log_z_sigma_tile, c)

        z_mu_tile = tf.expand_dims(z_mu, axis=-2)
        c = tf.constant([1, self.num_clusters, 1], tf.int32)
        z_mu_tile = tf.tile(z_mu_tile, c)

        c_sigma = tf.math.exp(self.log_c_sigma)
        p_z_c = tf.stack([tf.math.log(
            tfd.MultivariateNormalDiag(loc=self.c_mu[i, :], scale_diag=tf.math.sqrt(c_sigma[i, :])).prob(
                z_sample) + 1e-30) for i in range(self.num_clusters)], axis=-1)

        prior = self.prior

        p_c_z = tf.math.log(prior + tf.keras.backend.epsilon()) + p_z_c

        norm_s = tf.math.log(1e-30 + tf.math.reduce_sum(tf.math.exp(p_c_z), axis=-1, keepdims=True))
        c = tf.constant([1, self.num_clusters], tf.int32)
        norm = tf.tile(norm_s, c)
        p_c_z = tf.math.exp(p_c_z - norm)

        loss_1a = tf.math.log(c_sigma + tf.keras.backend.epsilon())

        loss_1b = tf.math.exp(log_z_sigma_tile) / (c_sigma + tf.keras.backend.epsilon())

        loss_1c = tf.math.square(z_mu_tile - self.c_mu) / (c_sigma + tf.keras.backend.epsilon())

        loss_1d = self.encoded_size * tf.math.log(tf.keras.backend.constant(2 * np.pi))

        loss_1a = tf.multiply(p_c_z, tf.math.reduce_sum(loss_1a, axis=-1))
        loss_1b = tf.multiply(p_c_z, tf.math.reduce_sum(loss_1b, axis=-1))
        loss_1c = tf.multiply(p_c_z, tf.math.reduce_sum(loss_1c, axis=-1))
        loss_1d = tf.multiply(p_c_z, loss_1d)

        loss_1a = 1 / 2 * tf.reduce_sum(loss_1a, axis=-1)
        loss_1b = 1 / 2 * tf.reduce_sum(loss_1b, axis=-1)
        loss_1c = 1 / 2 * tf.reduce_sum(loss_1c, axis=-1)
        loss_1d = 1 / 2 * tf.reduce_sum(loss_1d, axis=-1)

        loss_2a = - tf.math.reduce_sum(tf.math.xlogy(p_c_z, prior), axis=-1)

        if training:
            ind1, ind2, data = W
            ind1 = tf.reshape(ind1, [-1])
            ind2 = tf.reshape(ind2, [-1])
            data = tf.reshape(data, [-1])
            ind = tf.stack([ind1, ind2], axis=0)
            ind = tf.transpose(ind)
            ind = tf.dtypes.cast(ind, tf.int64)
            W_sparse = tf.SparseTensor(indices=ind, values=data, dense_shape=[len(inputs), len(inputs)])
            W_sparse = tf.sparse.expand_dims(W_sparse, axis=-1)
            W_tile = tf.sparse.concat(-1, [W_sparse] * self.num_clusters)
            mul = W_tile.__mul__(p_c_z)
            sum_j = tf.sparse.reduce_sum(mul, axis=-2)
            loss_2a_constrain = - tf.math.reduce_sum(tf.multiply(p_c_z, sum_j), axis=-1)

            self.add_loss(tf.math.reduce_mean(loss_2a_constrain))
            self.add_metric(loss_2a_constrain, name='loss_2a_c', aggregation="mean")

        loss_2b = tf.math.reduce_sum(tf.math.xlogy(p_c_z, p_c_z), axis=-1)

        loss_3 = - 1 / 2 * tf.reduce_sum(log_z_sigma + 1, axis=-1)

        self.add_loss(tf.math.reduce_mean(loss_1a))
        self.add_loss(tf.math.reduce_mean(loss_1b))
        self.add_loss(tf.math.reduce_mean(loss_1c))
        self.add_loss(tf.math.reduce_mean(loss_1d))
        self.add_loss(tf.math.reduce_mean(loss_2a))
        self.add_loss(tf.math.reduce_mean(loss_2b))
        self.add_loss(tf.math.reduce_mean(loss_3))
        self.add_metric(loss_1a, name='loss_1a', aggregation="mean")
        self.add_metric(loss_1b, name='loss_1b', aggregation="mean")
        self.add_metric(loss_1c, name='loss_1c', aggregation="mean")
        self.add_metric(loss_1d, name='loss_1d', aggregation="mean")
        self.add_metric(loss_2a, name='loss_2a', aggregation="mean")
        self.add_metric(loss_2b, name='loss_2b', aggregation="mean")
        self.add_metric(loss_3, name='loss_3', aggregation="mean")

        dec = self.decoder(z_sample)
        return dec, z_sample, p_z_c, p_c_z


def loss_DCGMM(x, x_decoded_mean):
    loss = 784 * tf.keras.losses.BinaryCrossentropy()(x, x_decoded_mean)
    return loss

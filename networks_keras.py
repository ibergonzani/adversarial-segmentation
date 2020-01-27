import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import InputLayer, Add, Concatenate, Conv2D, Dense, Reshape
from tensorflow.keras.layers import Flatten, LeakyReLU, BatchNormalization, MaxPool2D, Conv2DTranspose

import functools
import operator

def generator_net(input_shape):

	output_channels = input_shape[-1]

	input_layer = Input(shape=input_shape, name="Input")
	cv1 = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', name='conv9_i')(input_layer)
	cv1 = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', name='conv9_i')(input_layer)
	net = LeakyReLU()(net)

	net = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', name='conv3a_1')(net)
	net = BatchNormalization()(net)
	net = LeakyReLU()(net)

	net = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', name='conv3b_1')(net)
	res1 = BatchNormalization()(net)

	sum1 = Add()([cv1, res1])

	net = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', name='conv3a_2')(sum1)
	net = BatchNormalization()(net)
	net = LeakyReLU()(net)

	net = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', name='conv3b_2')(net)
	res2 = BatchNormalization()(net)

	sum2 = Add()([sum1, res2])

	net = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', name='conv3a_3')(sum2)
	net = BatchNormalization()(net)
	net = LeakyReLU()(net)

	net = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', name='conv3b_3')(net)
	res3 = BatchNormalization()(net)

	sum3 = Add()([sum2, res3])

	net = Conv2D(filters=output_channels, kernel_size=3, strides=(1, 1), padding='same', name='conv9_f')(sum3)
	output_layer =  tf.keras.activations.sigmoid(net)

	return Model(inputs=input_layer, outputs=output_layer, name="generator")




def discriminator_net(input_shape):

	input_layer = Input(shape=input_shape, name="Input")
	net = Conv2D(filters=32, kernel_size=5, strides=(1, 1), name='conv1')(input_layer)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)
	net = Conv2D(filters=64, kernel_size=3, strides=(1, 1), name='conv2')(net)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)
	net = MaxPool2D(pool_size=2, strides=(2, 2), padding='same', name='maxpool1')(net)
	net = Conv2D(filters=64, kernel_size=3, strides=(1, 1), name='conv3')(net)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)
	net = MaxPool2D(pool_size=2, strides=(2, 2), padding='same', name='maxpool2')(net)
	net = Conv2D(filters=64, kernel_size=3, strides=(1, 1), name='conv5')(net)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)
	net = MaxPool2D(pool_size=2, strides=(2, 2), padding='same', name='maxpool3')(net)
	net = Conv2D(filters=32, kernel_size=3, strides=(1, 1), name='conv7')(net)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)
	net = MaxPool2D(pool_size=2, strides=(2, 2), padding='same', name='maxpool4')(net)
	net = Flatten(name='flatten')(net)
	# net = dense(net, 1024, activation=tf.nn.leaky_relu, name='dense1')
	# net = batch_normalization(net)
	net = Dense(256, name='dense2')(net)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)
	net = Dense(256, name='dense3')(net)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)
	net = Dense(128, name='dense4')(net)
	net = LeakyReLU()(net)
	output_layer = Dense(1, activation=tf.nn.sigmoid, name='out')(net)

	return Model(inputs=input_layer, outputs=output_layer, name="discriminator_model")





def generator_bipath_net(input_shape):

	output_channels = input_shape[-1]

	# branch  1 ( color reconstruction)
	input_layer = Input(shape=input_shape, name="Input")
	cv1 = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv9_i')(input_layer)
	net = LeakyReLU()(cv1)
	net = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv3a_1')(net)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)
	net = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv3b_1')(net)
	res1 = BatchNormalization()(net)

	sum1  = Add()([cv1, res1])

	net = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv3a_2')(sum1)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)
	net = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv3b_2')(net)
	res2 = BatchNormalization()(net)

	sum2 = Add()([sum1, res2])

	br1 = Conv2D(filters=output_channels, kernel_size=3, strides=(1, 1), padding='same', name='conv9_f')(sum2)


	# branch 2 (features extraction)
	br2 = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv_bf1')(input_layer)
	br2 = LeakyReLU()(br2)
	br2 = BatchNormalization()(br2)
	br2 = MaxPool2D(pool_size=2, strides=(2, 2), name='maxpool1')(br2)
	br2 = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv_bf2')(br2)
	br2 = LeakyReLU()(br2)
	br2 = BatchNormalization()(br2)
	br2 = MaxPool2D(pool_size=2, strides=(2, 2), name='maxpool2a')(br2)
	br2 = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv_bf3')(br2)
	br2 = LeakyReLU()(br2)
	br2 = BatchNormalization()(br2)
	br2 = MaxPool2D(pool_size=2, strides=(2, 2), name='maxpool2')(br2)

	br2 = Conv2DTranspose(filters=16, kernel_size=3, padding='same', strides=(2, 2), name="deconv_1")(br2)
	br2 = LeakyReLU()(br2)
	br2 = BatchNormalization()(br2)
	br2 = Conv2DTranspose(filters=16, kernel_size=3, padding='same', strides=(2, 2), name="deconv_2")(br2)
	br2 = LeakyReLU()(br2)
	br2 = BatchNormalization()(br2)
	br2 = Conv2DTranspose(filters=16, kernel_size=3, padding='same', strides=(2, 2), name="deconv_3")(br2)
	br2 = LeakyReLU()(br2)
	br2 = BatchNormalization()(br2)

	# concatenate branches and reconstruct image
	net = Concatenate(axis=3)([br1, br2])
	net = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv9_f1')(net)
	net = LeakyReLU()(net)
	net = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv9_f2')(net)
	net = LeakyReLU()(net)

	output_layer = Conv2D(filters=output_channels, kernel_size=3, strides=(1, 1), padding='same', name='conv9_f3')(net)

	return Model(inputs=input_layer, outputs=output_layer, name="generator_model")




def encoder_decoder_net(input_shape, latent_dim):

	output_channels = input_shape[-1]

	############################### ENCODER MODEL ######################################################
	input_layer = Input(shape=input_shape)

	net = Conv2D(filters=40, kernel_size=3, strides=(1, 1), padding='same')(input_layer)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)
	net = MaxPool2D(pool_size=2, strides=(2, 2))(net)

	net = Conv2D(filters=80, kernel_size=3, strides=(1, 1), padding='same')(net)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)
	net = MaxPool2D(pool_size=2, strides=(2, 2))(net)

	net = Conv2D(filters=120, kernel_size=3, strides=(1, 1), padding='same')(net)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)
	net = MaxPool2D(pool_size=2, strides=(2, 2))(net)

	net = Conv2D(filters=160, kernel_size=3, strides=(1, 1), padding='same')(net)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)
	net = MaxPool2D(pool_size=2, strides=(2, 2))(net)

	net = Conv2D(filters=200, kernel_size=3, strides=(1, 1), padding='same')(net)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)
	net = MaxPool2D(pool_size=2, strides=(2, 2))(net)

	conv_shape = net.shape[-3:]
	conv_units = functools.reduce(operator.mul, conv_shape, 1)
	print("CONV_UNITS:", conv_units)

	net = Flatten()(net)
	net = Dense(conv_units / 2, activation=tf.nn.tanh)(net)
	net = Dense(latent_dim, activation=tf.nn.tanh)(net)

	encoder = Model(inputs=input_layer, outputs=net, name="encoder_net")

	#################### DECODER MODEL #########################################3
	input_layer = Input(shape=[latent_dim])

	net = Dense(conv_units / 2, activation=tf.nn.tanh)(input_layer)
	net = Dense(conv_units, activation=tf.nn.tanh)(net)
	net = Reshape(conv_shape)(net)

	net = Conv2DTranspose(filters=200, kernel_size=3, padding='same', strides=(2, 2))(net)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)

	net = Conv2DTranspose(filters=160, kernel_size=3, padding='same', strides=(2, 2))(net)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)

	net = Conv2DTranspose(filters=120, kernel_size=3, padding='same', strides=(2, 2))(net)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)

	net = Conv2DTranspose(filters=80, kernel_size=3, padding='same', strides=(2, 2))(net)
	net = LeakyReLU()(net)
	net = BatchNormalization()(net)

	net = Conv2DTranspose(filters=40, kernel_size=3, padding='same', strides=(2, 2))(net)
	net = LeakyReLU()(net)

	net = Conv2D(filters=output_channels, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.tanh)(net)

	decoder = Model(inputs=input_layer, outputs=net, name="decoder_net")

	return encoder, decoder





def autoencoder_net(input_shape):

	output_channels = input_shape[-1]

	# input_layer = Input(shape=input_shape, name="Input")
	# br2 = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv_bf1')(input_layer)
	# br2 = LeakyReLU()(br2)
	# br2 = BatchNormalization()(br2)
	# br2 = Conv2D(filters=16, kernel_size=3, strides=(2, 2), padding='valid', name='pool1')(br2)
	#
	# br2 = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv_bf2')(br2)
	# br2 = LeakyReLU()(br2)
	# br2 = BatchNormalization()(br2)
	# br2 = MaxPool2D(pool_size=2, strides=(2, 2), name='maxpool2a')(br2)
	#
	# br2 = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv_bf3')(br2)
	# br2 = LeakyReLU()(br2)
	# br2 = BatchNormalization()(br2)
	# br2 = MaxPool2D(pool_size=2, strides=(2, 2), name='maxpool2')(br2)
	#
	# shape = br2.shape[-3:]
	# units = functools.reduce(operator.mul, shape, 1)
	#
	# br2 = Flatten()(br2)
	# br2 = Dense(1024)(br2)
	# br2 = LeakyReLU()(br2)
	# br2 = Dense(units)(br2)
	# br2 = LeakyReLU()(br2)
	# br2 = Reshape(shape)(br2)
	#
	# br2 = Conv2DTranspose(filters=16, kernel_size=3, padding='same', strides=(2, 2), name="deconv_1")(br2)
	# br2 = LeakyReLU()(br2)
	# br2 = BatchNormalization()(br2)
	# br2 = Conv2DTranspose(filters=16, kernel_size=3, padding='same', strides=(2, 2), name="deconv_2")(br2)
	# br2 = LeakyReLU()(br2)
	# br2 = BatchNormalization()(br2)
	# br2 = Conv2DTranspose(filters=16, kernel_size=3, padding='same', strides=(2, 2), name="deconv_3")(br2)
	# br2 = LeakyReLU()(br2)

	input_layer = Input(shape=input_shape, name="Input")
	br2 = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv_bf1')(input_layer)
	br2 = LeakyReLU()(br2)
	br2 = BatchNormalization()(br2)
	br2 = MaxPool2D(pool_size=2, strides=(2, 2), name='maxpool1')(br2)
	br2 = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv_bf2')(br2)
	br2 = LeakyReLU()(br2)
	br2 = BatchNormalization()(br2)
	br2 = MaxPool2D(pool_size=2, strides=(2, 2), name='maxpool2a')(br2)
	br2 = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv_bf3')(br2)
	br2 = LeakyReLU()(br2)
	br2 = BatchNormalization()(br2)
	br2 = MaxPool2D(pool_size=2, strides=(2, 2), name='maxpool2')(br2)

	shape = br2.shape[-3:]
	units = functools.reduce(operator.mul, shape, 1)

	br2 = Flatten()(br2)
	br2 = Dense(4096)(br2)
	br2 = LeakyReLU()(br2)
	br2 = Dense(units)(br2)
	br2 = LeakyReLU()(br2)
	br2 = Reshape(shape)(br2)

	br2 = Conv2DTranspose(filters=16, kernel_size=3, padding='same', strides=(2, 2), name="deconv_1")(br2)
	br2 = LeakyReLU()(br2)
	br2 = BatchNormalization()(br2)
	br2 = Conv2DTranspose(filters=16, kernel_size=3, padding='same', strides=(2, 2), name="deconv_2")(br2)
	br2 = LeakyReLU()(br2)
	br2 = BatchNormalization()(br2)
	br2 = Conv2DTranspose(filters=16, kernel_size=3, padding='same', strides=(2, 2), name="deconv_3")(br2)
	br2 = LeakyReLU()(br2)

	output_layer = Conv2D(filters=output_channels, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.tanh, name='conv9_f3')(br2)

	return Model(inputs=input_layer, outputs=output_layer, name="autoencoder_model")

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import InputLayer, Add, Concatenate, Conv2D, Dense, Flatten, LeakyReLU, BatchNormalization, MaxPool2D, Conv2DTranspose

def generator_net(input_shape):
	
	output_channels = input_shape[-1]
		
	input_layer = Input(shape=input_shape, name="Input")  
	cv1 = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', name='conv9_i')(input_layer)
	cv1 = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', name='conv9_i')(input_layer)
	net = LeakyReLU()(net)
	
	net = Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same', name='conv3a_1')(net)
	net = BatchNormalization()(net)
	net = LeakyReLU()(net)
	
	net = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', name='conv3b_1')(net)
	res1 = BatchNormalization()(net)
	
	sum1 = Add()([cv1, res1])
	
	net = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', name='conv3a_2')(sum1)
	net = BatchNormalization()(net)
	net = LeakyReLU()(net)
	
	net = Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same', name='conv3b_2')(net)
	res2 = BatchNormalization()(net)
	
	sum2 = Add()([sum1, res2])
	
	net = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', name='conv3a_3')(sum2)
	net = BatchNormalization()(net)
	net = LeakyReLU()(net)
	
	net = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same', name='conv3b_3')(net)
	res3 = BatchNormalization()(net)
	
	sum3 = Add()([sum2, res3])

	output_layer = Conv2D(filters=output_channels, kernel_size=3, strides=(1, 1), padding='same', name='conv9_f')(sum3)
	
	return Model(inputs=input_layer, outputs=output_layer, name="generator")
	



def discriminator_net(input_shape):
	
	input_layer = Input(shape=input_shape, name="Input")  
	net = Conv2D(filters=32, kernel_size=5, strides=(1, 1), name='conv1')(input_layer)
	net = LeakyReLU()(net)
	net = Conv2D(filters=64, kernel_size=3, strides=(1, 1), name='conv2')(net)
	net = LeakyReLU()(net)
	net = MaxPool2D(pool_size=2, strides=(2, 2), padding='same', name='maxpool1')(net)
	net = Conv2D(filters=64, kernel_size=3, strides=(1, 1), name='conv3')(net)
	net = LeakyReLU()(net)
	net = MaxPool2D(pool_size=2, strides=(2, 2), padding='same', name='maxpool2a')(net)
	net = Conv2D(filters=64, kernel_size=3, strides=(1, 1), name='conv4')(net)
	net = LeakyReLU()(net)
	net = MaxPool2D(pool_size=2, strides=(2, 2), padding='same', name='maxpool2')(net)
	net = Conv2D(filters=64, kernel_size=3, strides=(1, 1), name='conv5')(net)
	net = LeakyReLU()(net)
	net = MaxPool2D(pool_size=2, strides=(2, 2), padding='same', name='maxpool3a')(net)
	net = Conv2D(filters=64, kernel_size=3, strides=(1, 1), name='conv6')(net)
	net = LeakyReLU()(net)
	net = MaxPool2D(pool_size=2, strides=(2, 2), padding='same', name='maxpool3')(net)
	net = Conv2D(filters=32, kernel_size=3, strides=(1, 1), name='conv7')(net)
	net = LeakyReLU()(net)
	net = MaxPool2D(pool_size=2, strides=(2, 2), padding='same', name='maxpool4')(net)
	net = Flatten(name='flatten')(net)
	# net = dense(net, 1024, activation=tf.nn.leaky_relu, name='dense1')
	# net = batch_normalization(net)
	net = Dense(512, name='dense2')(net)
	net = LeakyReLU()(net)
	# net = batch_normalization(net)
	net = Dense(256, name='dense3')(net)
	net = LeakyReLU()(net)
	# net = batch_normalization(net)
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
	net = Conv2D(filters=16, kernel_size=5, strides=(1, 1), padding='same', name='conv3a_1')(net)
	net = BatchNormalization()(net)
	net = LeakyReLU()(net)
	net = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv3b_1')(net)
	res1 = BatchNormalization()(net)
	
	sum1  = Add()([cv1, res1])
	
	net = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv3a_2')(sum1)
	net = BatchNormalization()(net)
	net = LeakyReLU()(net)
	net = Conv2D(filters=16, kernel_size=5, strides=(1, 1), padding='same', name='conv3b_2')(net)
	res2 = BatchNormalization()(net)
	
	sum2 = Add()([sum1, res2])

	br1 = Conv2D(filters=output_channels, kernel_size=3, strides=(1, 1), padding='same', name='conv9_f')(sum2)
	
	
	# branch 2 (features extraction)
	br2 = Conv2D(filters=16, kernel_size=5, strides=(1, 1), padding='same', name='conv_bf1')(input_layer)
	net = LeakyReLU()(net)
	br2 = MaxPool2D(pool_size=2, strides=(2, 2), name='maxpool1')(br2)
	br2 = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv_bf2')(br2)
	net = LeakyReLU()(net)
	br2 = MaxPool2D(pool_size=2, strides=(2, 2), name='maxpool2a')(br2)
	br2 = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv_bf3')(br2)
	net = LeakyReLU()(net)
	br2 = MaxPool2D(pool_size=2, strides=(2, 2), name='maxpool2')(br2)
	
	br2 = Conv2DTranspose(filters=16, kernel_size=3, padding='same', strides=(2, 2), name="deconv_1")(br2)
	net = LeakyReLU()(net)
	br2 = Conv2DTranspose(filters=16, kernel_size=3, padding='same', strides=(2, 2), name="deconv_2")(br2)
	net = LeakyReLU()(net)
	br2 = Conv2DTranspose(filters=16, kernel_size=3, padding='same', strides=(2, 2), name="deconv_3")(br2)
	net = LeakyReLU()(net)
	
	# concatenate branches and reconstruct image
	net = Concatenate(axis=3)([br1, br2])
	net = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same', name='conv9_f1')(net)
	net = LeakyReLU()(net)
	
	output_layer = Conv2D(filters=output_channels, kernel_size=3, strides=(1, 1), padding='same', name='conv9_f2')(net)
	
	return Model(inputs=input_layer, outputs=output_layer, name="generator_model")
	

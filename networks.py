import tensorflow as tf
from tensorflow.layers import conv2d, conv2d_transpose, dense, flatten, batch_normalization, max_pooling2d
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.nn import relu, leaky_relu

def convolutional_autoencoder_net(inputs, scope, reuse=None, rgb=False):
	
	output_channels = 3 if rgb else 1
	
	with tf.variable_scope(scope, reuse=reuse):
		
		cv1   = conv2d(inputs, filters=32, kernel_size=3, strides=1, padding='same', activation=None, name='conv9_i')
		cv1_r = leaky_relu(cv1)
		
		res1_c = conv2d(cv1_r, filters=32, kernel_size=5, strides=1, padding='same', activation=None, name='conv3a_1')
		res1_b = batch_normalization(res1_c)
		res1_r = leaky_relu(res1_b)
		
		res1_d = conv2d(res1_r, filters=32, kernel_size=3, strides=1, padding='same', activation=None, name='conv3b_1')
		res1   = batch_normalization(res1_d)
		
		sum1  = cv1 + res1
		
		res2_c = conv2d(sum1, filters=32, kernel_size=3, strides=1, padding='same', activation=None, name='conv3a_2')
		res2_b = batch_normalization(res2_c)
		res2_r = leaky_relu(res2_b)
		
		res2_d = conv2d(res2_r, filters=32, kernel_size=5, strides=1, padding='same', activation=None, name='conv3b_2')
		res2   = batch_normalization(res2_d)
		
		sum2 = sum1 + res2
		
		res3_c = conv2d(sum2, filters=32, kernel_size=3, strides=1, padding='same', activation=None, name='conv3a_3')
		res3_b = batch_normalization(res3_c)
		res3_r = leaky_relu(res3_b)
		
		res3_d = conv2d(res3_r, filters=32, kernel_size=3, strides=1, padding='same', activation=None, name='conv3b_3')
		res3   = batch_normalization(res3_d)
		
		sum3 = sum2 + res3

		model = conv2d(sum3, filters=output_channels, kernel_size=3, strides=1, padding='same', activation=None, name='conv9_f')
		
		return model
	
	



def classifier_net(inputs, scope, reuse=None):
	
	with tf.variable_scope(scope, reuse=reuse):
		
		net = conv2d(inputs, filters=32, kernel_size=5, strides=1, activation=tf.nn.leaky_relu, name='conv1')
		net = conv2d(net, filters=64, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name='conv2')
		net = max_pooling2d(net, pool_size=2, strides=2, padding='same', name='maxpool1')
		net = conv2d(net, filters=64, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name='conv3')
		net = max_pooling2d(net, pool_size=2, strides=2, padding='same', name='maxpool2a')
		net = conv2d(net, filters=64, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name='conv4')
		net = max_pooling2d(net, pool_size=2, strides=2, padding='same', name='maxpool2')
		net = conv2d(net, filters=64, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name='conv5')
		net = max_pooling2d(net, pool_size=2, strides=2, padding='same', name='maxpool3a')
		net = conv2d(net, filters=64, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name='conv6')
		net = max_pooling2d(net, pool_size=2, strides=2, padding='same', name='maxpool3')
		net = conv2d(net, filters=32, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name='conv7')
		net = max_pooling2d(net, pool_size=2, strides=2, padding='same', name='maxpool4')
		net = flatten(net, name='flatten')
		# net = dense(net, 1024, activation=tf.nn.leaky_relu, name='dense1')
		# net = batch_normalization(net)
		net = dense(net, 512, activation=tf.nn.leaky_relu, name='dense2')
		# net = batch_normalization(net)
		net = dense(net, 256, activation=tf.nn.leaky_relu, name='dense3')
		# net = batch_normalization(net)
		net = dense(net, 128, activation=tf.nn.leaky_relu, name='dense4')
		net = dense(net, 1, activation=tf.nn.sigmoid, name='out')

		return net

		
		
		
		
def generator_net(inputs, scope, reuse=None, rgb=False):
	
	output_channels = 3 if rgb else 1
	
	with tf.variable_scope(scope, reuse=reuse):
	
		# branch  1 ( color reconstruction)
		
		cv1   = conv2d(inputs, filters=16, kernel_size=3, strides=1, padding='same', activation=None, name='conv9_i')
		cv1_r = leaky_relu(cv1)
		
		res1_c = conv2d(cv1_r, filters=16, kernel_size=5, strides=1, padding='same', activation=None, name='conv3a_1')
		res1_b = batch_normalization(res1_c)
		res1_r = leaky_relu(res1_b)
		
		res1_d = conv2d(res1_r, filters=16, kernel_size=3, strides=1, padding='same', activation=None, name='conv3b_1')
		res1   = batch_normalization(res1_d)
		
		sum1  = cv1 + res1
		
		res2_c = conv2d(sum1, filters=16, kernel_size=3, strides=1, padding='same', activation=None, name='conv3a_2')
		res2_b = batch_normalization(res2_c)
		res2_r = leaky_relu(res2_b)
		
		res2_d = conv2d(res2_r, filters=16, kernel_size=3, strides=1, padding='same', activation=None, name='conv3b_2')
		res2   = batch_normalization(res2_d)
		
		br1 = sum1 + res2
		
		
		# branch 2 (features extraction)
		br2 = conv2d(inputs, filters=16, kernel_size=5, strides=1, padding='same', activation=tf.nn.leaky_relu, name='conv_bf1')
		br2 = max_pooling2d(br2, pool_size=2, strides=2, name='maxpool1')
		br2 = conv2d(br2, filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, name='conv_bf2')
		br2 = max_pooling2d(br2, pool_size=2, strides=2, name='maxpool2a')
		br2 = conv2d(br2, filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, name='conv_bf3')
		br2 = max_pooling2d(br2, pool_size=2, strides=2, name='maxpool2')
		
		print(br2.shape)
		br2 = conv2d_transpose(br2, filters=16, kernel_size=3, padding='same', strides=2, activation=tf.nn.leaky_relu, name="deconv_1")
		print(br2.shape)
		br2 = conv2d_transpose(br2, filters=16, kernel_size=3, padding='same', strides=2, activation=tf.nn.leaky_relu, name="deconv_2")
		print(br2.shape)
		br2 = conv2d_transpose(br2, filters=16, kernel_size=3, padding='same', strides=2, activation=tf.nn.leaky_relu, name="deconv_3")
		print(br2.shape)
		
		# concatenate branches and reconstruct image
		sum3 = tf.concat((br1, br2), axis=3);
		model = conv2d(sum3, filters=output_channels, kernel_size=3, strides=1, padding='same', activation=None, name='conv9_f')
		
		return model
		

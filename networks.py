import tensorflow as tf
from tensorflow.layers import conv2d, dense, flatten, conv2d_transpose, batch_normalization
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.nn import relu, leaky_relu

def convolutional_autoencoder_net(inputs, scope, reuse=None):
	
	with tf.variable_scope(scope, reuse=reuse):
		
		cv1   = conv2d(inputs, filters=32, kernel_size=9, strides=1, padding='same', activation=None, name='conv9_i')
		cv1_r = leaky_relu(cv1)
		
		res1_c = conv2d(cv1_r, filters=32, kernel_size=3, strides=1, padding='same', activation=None, name='conv3a_1')
		res1_b = batch_normalization(res1_c)
		res1_r = leaky_relu(res1_b)
		
		res1_d = conv2d(res1_r, filters=32, kernel_size=3, strides=1, padding='same', activation=None, name='conv3b_1')
		res1   = batch_normalization(res1_d)
		
		sum1  = cv1 + res1
		
		res2_c = conv2d(sum1, filters=32, kernel_size=3, strides=1, padding='same', activation=None, name='conv3a_2')
		res2_b = batch_normalization(res2_c)
		res2_r = leaky_relu(res2_b)
		
		res2_d = conv2d(res2_r, filters=32, kernel_size=3, strides=1, padding='same', activation=None, name='conv3b_2')
		res2   = batch_normalization(res2_d)
		
		sum2 = sum1 + res2
		
		res3_c = conv2d(sum2, filters=32, kernel_size=3, strides=1, padding='same', activation=None, name='conv3a_3')
		res3_b = batch_normalization(res3_c)
		res3_r = leaky_relu(res3_b)
		
		res3_d = conv2d(res3_r, filters=32, kernel_size=3, strides=1, padding='same', activation=None, name='conv3b_3')
		res3   = batch_normalization(res3_d)
		
		sum3 = sum2 + res3

		model = conv2d(sum3, filters=1, kernel_size=3, strides=1, padding='same', activation=None, name='conv9_f')
		
		return model
	


def classifier_net(inputs, scope, reuse=None):
	
	with tf.variable_scope(scope, reuse=reuse):
		
		cv1 = conv2d(inputs, filters=32, kernel_size=4, strides=2, activation=tf.nn.relu, name='conv1')
		cv2 = conv2d(cv1, filters=32, kernel_size=4, strides=1, activation=tf.nn.relu, name='conv2')
		fl = flatten(cv2, name='flatten')
		d1 = dense(fl, 1024, activation=tf.nn.relu, name='dense1')
		d2 = dense(d1, 512, activation=tf.nn.relu, name='dense2')
		d3 = dense(d2, 128, activation=tf.nn.relu, name='dense3')
		model = dense(d3, 1, activation=None, name='dense4')
		# return model
		# model = Sequential()
		# model.add(InputLayer(input_shape=inputs.shape[1:], input_tensor=inputs))
		# model.add(Conv2D(filters=32, kernel_size=4, strides=2))
		# model.add(Conv2D(filters=16, kernel_size=4, strides=1))
		# model.add(Flatten())
		# model.add(Dense(1024, activation=tf.nn.relu))
		# model.add(Dense(512, activation=tf.nn.relu))
		# model.add(Dense(128, activation=tf.nn.relu))
		# model.add(Dense(1))
		#model.add(Activation('sigmoid'))
		return model

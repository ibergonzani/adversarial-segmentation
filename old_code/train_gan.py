import tensorflow as tf
import progressbar as pb
import numpy as np
import networks
import math
import time
import cv2
import os

from phantom_dataset import load_dataset_separated
from mri_dataset import load_stare_dataset_separated
from mri_dataset import load_messidor_dataset_binary
from mias_dataset import load_mias_dataset
from bhi_dataset import load_bhi_dataset
from brain_dataset import load_brain_dataset


timestamp = time.time()

DATASET_FOLDER = 'phantom/'
TRAINING_FOLDER = DATASET_FOLDER + 'train/'
VALIDATION_FOLDER = DATASET_FOLDER + 'val/'

TRAIN_LOG_FOLDER = 'logs/train_' + str(timestamp)
TEST_LOG_FOLDER  = 'logs/test_' + str(timestamp)

EPOCHS = 4
STEPSIZE = 1e-4
BATCH_SIZE = 1
LR_DECAY = 1.0
AE_LR_DECAY = 2.0



# loading and standardize data
# x_train_healthy, x_train_disease = load_dataset_separated(TRAINING_FOLDER) 
# x_test_healthy,  x_test_disease  = load_dataset_separated(VALIDATION_FOLDER)

# x_train_healthy, x_train_disease = load_stare_dataset_separated()
# x_test_healthy,  x_test_disease  = load_stare_dataset_separated()

print("Loading Dataset")
Xh, Xu = load_brain_dataset() # load_messidor_dataset_binary(1)
print("Number of negative and positive samples:", Xh.shape, Xu.shape)

# shuffling data
np.random.seed(1994)
np.random.shuffle(Xh)
np.random.shuffle(Xu)

# take same number of positive and negative examples
min_size = min(Xh.shape[0], Xu.shape[0])
Xh = Xh[:min_size, ...]
Xu = Xu[:min_size, ...]

# splitting sets in training and test data
size_ht = int(Xh.shape[0] * 0.8)		# size of healthy images training set
size_ut = int(Xu.shape[0] * 0.8)		# size of unhealthy images training set
print("healthy-unhealthy training set sizes:", size_ht, size_ut)

x_train_healthy, x_train_disease = Xh[:size_ht , ...], Xu[:size_ut , ...]
x_test_healthy,  x_test_disease  = Xh[ size_ht:, ...], Xu[ size_ut:, ...]

print(x_train_healthy.shape, x_test_healthy.shape)
print(x_train_disease.shape, x_test_disease.shape)

print(type(x_train_healthy), type(x_test_healthy))
print(type(x_train_disease), type(x_test_disease))

# data normalization
xmean = 127.5
xstd = 127.5

x_train_healthy   = (x_train_healthy - xmean) / xstd
x_train_disease  = (x_train_disease - xmean) / xstd

x_test_healthy    = (x_test_healthy  - xmean) / xstd
x_test_disease   = (x_test_disease  - xmean) / xstd


# dataset preparation using tensorflow dataset iterators
batch_size = tf.placeholder(tf.int64)

####################### GENERATOR Dataset
data_features = tf.placeholder(tf.float32, (None,) + x_train_healthy.shape[1:])

train_data = tf.data.Dataset.from_tensor_slices(data_features)
train_data = train_data.repeat().shuffle(x_train_disease.shape[0]).batch(batch_size)

test_data = tf.data.Dataset.from_tensor_slices(data_features)
test_data = test_data.repeat().batch(batch_size)

data_iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

features = data_iterator.get_next()
train_initialization = data_iterator.make_initializer(train_data)
test_initialization  = data_iterator.make_initializer(test_data)


####################### CLASSIFIER DATASET
data_features_healthy = tf.placeholder(tf.float32, (None,) + x_train_healthy.shape[1:])

train_data_healthy = tf.data.Dataset.from_tensor_slices(data_features_healthy)
train_data_healthy = train_data_healthy.repeat().shuffle(x_train_healthy.shape[0]).batch(batch_size)

test_data_healthy = tf.data.Dataset.from_tensor_slices(data_features_healthy)
test_data_healthy = test_data_healthy.repeat().batch(batch_size)

data_iterator_healthy = tf.data.Iterator.from_structure(train_data_healthy.output_types, train_data_healthy.output_shapes)

features_healthy = data_iterator_healthy.get_next()
train_initialization_healthy = data_iterator_healthy.make_initializer(train_data_healthy)
test_initialization_healthy  = data_iterator_healthy.make_initializer(test_data_healthy)


########################## NETWORKS ####################################

# image autoencoder network initialization (start from images with disease)
autoencoder = networks.generator_net(features, scope="generator", rgb=False)
ae_xnet, ae_ynet = features, autoencoder

# classification network initialization (for real images, healthy)
classifier = networks.classifier_net(features_healthy, scope="classifier")
xnet, ynet = features_healthy, classifier

# classification network initialization (for generated images)
gen_classifier = networks.classifier_net(ae_ynet, scope="classifier", reuse=True)
gen_xnet, gen_ynet = ae_ynet, gen_classifier


# segmentations
difference = ae_ynet - ae_xnet


############################## LOSSES ################################

mse_truth     = tf.losses.mean_squared_error(predictions=ynet, labels=tf.zeros_like(ynet))
mse_generated = tf.losses.mean_squared_error(predictions=gen_ynet, labels=tf.ones_like(gen_ynet)) #labels
loss = mse_truth + mse_generated

ae_loss_similarity = tf.losses.mean_squared_error(predictions=ae_ynet, labels=ae_xnet)
ae_loss_classification = tf.losses.mean_squared_error(predictions=gen_ynet, labels=tf.zeros_like(gen_ynet))
ae_loss = ae_loss_similarity + ae_loss_classification


########################### OPTIMIZERS #################################

gen_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
cla_vars = [var for var in tf.trainable_variables() if 'classifier' in var.name]


with tf.name_scope('trainer_optimizer'):
	learning_rate = tf.Variable(STEPSIZE, name='learning_rate')
	learning_rate_decay = tf.placeholder(tf.float32, shape=(), name='lr_decay')
	update_learning_rate = tf.assign(learning_rate, learning_rate / learning_rate_decay)
	
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		global_step = tf.train.get_or_create_global_step()
		train_op = optimizer.minimize(loss=loss, var_list=cla_vars, global_step=global_step)
		

with tf.name_scope('ae_trainer_optimizer'):
	ae_learning_rate = tf.Variable(STEPSIZE, name='learning_rate')
	ae_learning_rate_decay = tf.placeholder(tf.float32, shape=(), name='lr_decay')
	ae_update_learning_rate = tf.assign(learning_rate, ae_learning_rate / ae_learning_rate_decay)
	
	ae_optimizer = tf.train.AdamOptimizer(learning_rate=ae_learning_rate)
	
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		ae_global_step = tf.train.get_or_create_global_step()
		ae_train_op = ae_optimizer.minimize(loss=ae_loss, var_list=gen_vars, global_step=ae_global_step)

	
############################## METRICS ####################################

# metrics definition (classification)
with tf.variable_scope('metrics'):
	mloss, mloss_update	 = tf.metrics.mean(loss)
	accuracy, acc_update = tf.metrics.accuracy(tf.zeros_like(ynet), tf.round(ynet))

	metrics = [mloss, accuracy]
	metrics_update = [mloss_update, acc_update]
	
#metrics definition (autoencoder)
with tf.variable_scope('metrics'):
	ae_mloss, ae_mloss_update = tf.metrics.mean(ae_loss)

	ae_metrics = [ae_mloss]
	ae_metrics_update = [ae_mloss_update]
	
# Isolate the variables stored behind the scenes by the metric operation
metrics_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
metrics_initializer = tf.variables_initializer(metrics_variables)


# summaries
los_sum = tf.summary.scalar('loss', mloss)
acc_sum = tf.summary.scalar('accuracy', accuracy)
ae_loss_sum = tf.summary.scalar('ae_oss', ae_mloss)
merged_summary = tf.summary.merge([los_sum, acc_sum, ae_loss_sum])


# network weights saver
saver = tf.train.Saver()

NUM_BATCHES_TRAIN = math.ceil(x_train_healthy.shape[0] / BATCH_SIZE)
NUM_BATCHES_TEST  = math.ceil(x_test_healthy.shape[0]  / BATCH_SIZE)


# dynamic memory allocation
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.log_device_placement = False

with tf.Session(config=configuration) as sess:

	# tensorboard summary writer
	train_writer = tf.summary.FileWriter(TRAIN_LOG_FOLDER, sess.graph)
	test_writer  = tf.summary.FileWriter(TEST_LOG_FOLDER)
	
	sess.run(tf.global_variables_initializer())
	
	for epoch in range(EPOCHS):
		
		print("\nEPOCH %d/%d" % (epoch+1, EPOCHS))
		
		# exponential learning rate decay
		# if (epoch + 1) % 10 == 0:
			# sess.run(update_learning_rate, feed_dict={learning_rate_decay: LR_DECAY})
			# sess.run(ae_update_learning_rate, feed_dict={ae_learning_rate_decay: AE_LR_DECAY})
		
		
		# initialize training dataset and set batch normalization training
		sess.run(train_initialization, feed_dict={data_features:x_train_disease, batch_size:BATCH_SIZE})
		sess.run(train_initialization_healthy, feed_dict={data_features_healthy:x_train_healthy, batch_size:BATCH_SIZE})
		sess.run(metrics_initializer)
		
		progress_info = pb.ProgressBar(total=NUM_BATCHES_TRAIN, prefix=' train', show=True)
		
		# Training of the network
		for nb in range(NUM_BATCHES_TRAIN):
			_, out, _ = sess.run([train_op, ae_ynet, ae_train_op])	# train network on a single batch
			[batch_trn_loss, _], _ = sess.run([metrics_update, ae_metrics_update])
			[trn_loss, a], [ae_trn_loss] = sess.run([metrics, ae_metrics])
				
			progress_info.update_and_show( suffix = '  loss {:.4f},  acc: {:.3f}'.format(ae_trn_loss, a) )
		print()
		
		summary = sess.run(merged_summary)
		train_writer.add_summary(summary, epoch)
		
		
		
		# initialize the test dataset and set batc normalization inference
		sess.run(test_initialization, feed_dict={data_features:x_test_disease, batch_size:BATCH_SIZE})
		sess.run(test_initialization_healthy, feed_dict={data_features_healthy:x_test_healthy, batch_size:BATCH_SIZE})
		sess.run(metrics_initializer)
		
		progress_info = pb.ProgressBar(total=NUM_BATCHES_TEST, prefix='  eval', show=True)
		
		# evaluation of the network
		for nb in range(NUM_BATCHES_TEST):
			ins, _, out, _, _, _, dif = sess.run([ae_xnet, loss, ae_ynet, ae_loss, metrics_update, ae_metrics_update, difference])
			[val_loss, a], [ae_val_loss] = sess.run([metrics, ae_metrics])
						
			if (epoch + 1) % 1 == 0:
				out_dir = os.path.join("out/", str(epoch+1))
				if not os.path.exists(out_dir):
					os.makedirs(out_dir)
					
				ins = (ins * xstd) + xmean
				out = (out * xstd) + xmean
				dif = (dif * xstd) + xmean
			
				for i in range(out.shape[0]):
					ins_image = ins[i,:,:,:]
					out_image = out[i,:,:,:]
					
					seg_image = np.max(np.abs(ins_image - out_image), axis=2, keepdims=True)
					seg_image[seg_image >= 30] = 255
					seg_image[seg_image <  30] = 0
					
					origi_name = "image_" + '{:04d}'.format(i + nb*BATCH_SIZE) + "o.png"
					image_name = "image_" + '{:04d}'.format(i + nb*BATCH_SIZE) + "r.png"
					segme_name = "image_" + '{:04d}'.format(i + nb*BATCH_SIZE) + "segm.png"
					cv2.imwrite(os.path.join(out_dir, origi_name), ins_image)
					cv2.imwrite(os.path.join(out_dir, image_name), out_image)
					cv2.imwrite(os.path.join(out_dir, segme_name), seg_image)
				
				saver.save(sess, os.path.join("models", 'model.ckpt'), global_step=epoch+1)
			
			progress_info.update_and_show( suffix = '  loss {:.4f},  acc: {:.3f}'.format(ae_val_loss, a) )
		print()
		
		summary  = sess.run(merged_summary)
		test_writer.add_summary(summary, epoch)
		
	
	train_writer.close()
	test_writer.close()
	
	saver.save(sess, os.path.join("models", 'model.ckpt'))

#print('\nTraining completed!\nNetwork model is saved in  {}\nTraining logs are saved in {}'.format(session_modeldir, session_logdir))

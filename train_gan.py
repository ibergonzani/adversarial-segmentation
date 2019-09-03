import tensorflow as tf
import progressbar as pb
import numpy as np
import networks
import math
import time
import cv2
import os

from phantom_dataset import load_dataset_separated


timestamp = time.time()

DATASET_FOLDER = 'phantom/'
TRAINING_FOLDER = DATASET_FOLDER + 'train/'
VALIDATION_FOLDER = DATASET_FOLDER + 'val/'

TRAIN_LOG_FOLDER = 'logs/train_' + str(timestamp)
TEST_LOG_FOLDER  = 'logs/test_' + str(timestamp)

EPOCHS = 30
STEPSIZE = 1e-4
BATCH_SIZE = 4
LR_DECAY = 1.0
AE_LR_DECAY = 2.0



# loading and standardize data
x_train_healty, x_train_disease = load_dataset_separated(TRAINING_FOLDER) 
x_test_healty,  x_test_disease  = load_dataset_separated(VALIDATION_FOLDER)

x_full = np.concatenate((x_train_healty, x_train_disease), axis=0)
xmean = np.mean(x_full, axis=(0), keepdims=True)
xstd = np.std(x_full, axis=(0), keepdims=True)

x_train_healty   = (x_train_healty - xmean) / xstd
x_train_disease  = (x_train_disease - xmean) / xstd

x_test_healty    = (x_test_healty  - xmean) / xstd
x_test_disease   = (x_test_disease  - xmean) / xstd


# dataset preparation using tensorflow dataset iterators
batch_size    = tf.placeholder(tf.int64)

####################### GENERATOR Dataset
data_features = tf.placeholder(tf.float32, (None,) + x_full.shape[1:])

train_data = tf.data.Dataset.from_tensor_slices(data_features)
train_data = train_data.repeat().shuffle(x_train_disease.shape[0]).batch(batch_size)

test_data = tf.data.Dataset.from_tensor_slices(data_features)
test_data = test_data.repeat().batch(batch_size)

data_iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

features = data_iterator.get_next()
train_initialization = data_iterator.make_initializer(train_data)
test_initialization  = data_iterator.make_initializer(test_data)


####################### CLASSIFIER DATASET
data_features_healty = tf.placeholder(tf.float32, (None,) + x_train_healty.shape[1:])

train_data_healty = tf.data.Dataset.from_tensor_slices(data_features_healty)
train_data_healty = train_data_healty.repeat().shuffle(x_train_healty.shape[0]).batch(batch_size)

test_data_healty = tf.data.Dataset.from_tensor_slices(data_features_healty)
test_data_healty = test_data_healty.repeat().batch(batch_size)

data_iterator_healty = tf.data.Iterator.from_structure(train_data_healty.output_types, train_data_healty.output_shapes)

features_healty = data_iterator_healty.get_next()
train_initialization_healty = data_iterator_healty.make_initializer(train_data_healty)
test_initialization_healty  = data_iterator_healty.make_initializer(test_data_healty)


########################## NETWORKS ####################################

# image autoencoder network initialization (start from images with disease)
autoencoder = networks.convolutional_autoencoder_net(features, scope="generator")
ae_xnet, ae_ynet = features, autoencoder

# classification network initialization (for real images, healty)
classifier = networks.classifier_net(features_healty, scope="classifier")
xnet, ynet = features_healty, classifier

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

NUM_BATCHES_TRAIN = math.ceil(x_train_healty.shape[0] / BATCH_SIZE)
NUM_BATCHES_TEST  = math.ceil(x_test_healty.shape[0]  / BATCH_SIZE)

with tf.Session() as sess:

	# tensorboard summary writer
	train_writer = tf.summary.FileWriter(TRAIN_LOG_FOLDER, sess.graph)
	test_writer  = tf.summary.FileWriter(TEST_LOG_FOLDER)
	
	sess.run(tf.global_variables_initializer())
	
	for epoch in range(EPOCHS):
		
		print("\nEPOCH %d/%d" % (epoch+1, EPOCHS))
		
		# exponential learning rate decay
		if (epoch + 1) % 10 == 0:
			sess.run(update_learning_rate, feed_dict={learning_rate_decay: LR_DECAY})
			sess.run(ae_update_learning_rate, feed_dict={ae_learning_rate_decay: AE_LR_DECAY})
		
		
		# initialize training dataset and set batch normalization training
		sess.run(train_initialization, feed_dict={data_features:x_train_disease, batch_size:BATCH_SIZE})
		sess.run(train_initialization_healty, feed_dict={data_features_healty:x_train_healty, batch_size:BATCH_SIZE})
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
		sess.run(test_initialization_healty, feed_dict={data_features_healty:x_test_healty, batch_size:BATCH_SIZE})
		sess.run(metrics_initializer)
		
		progress_info = pb.ProgressBar(total=NUM_BATCHES_TEST, prefix='  eval', show=True)
		
		# evaluation of the network
		for nb in range(NUM_BATCHES_TEST):
			ins, _, out, _, _, _, dif = sess.run([ae_xnet, loss, ae_ynet, ae_loss, metrics_update, ae_metrics_update, difference])
			[val_loss, a], [ae_val_loss] = sess.run([metrics, ae_metrics])
			
			ins = (ins * xstd) + xmean
			out = (out * xstd) + xmean
			dif = (dif * xstd) + xmean
		
						
			if epoch == EPOCHS - 1:
				for i in range(out.shape[0]):
					ins_image = ins[i,:,:,:]
					out_image = out[i,:,:,:]
					
					seg_image = np.abs(ins_image - out_image)
					seg_image[seg_image >= 30] = 255
					seg_image[seg_image <  30] = 0
					
					origi_name = "out/image_" + '{:04d}'.format(i + nb*BATCH_SIZE) + "o.png"
					image_name = "out/image_" + '{:04d}'.format(i + nb*BATCH_SIZE) + "r.png"
					segme_name = "out/image_" + '{:04d}'.format(i + nb*BATCH_SIZE) + "segm.png"
					cv2.imwrite(origi_name, ins_image)
					cv2.imwrite(image_name, out_image)
					cv2.imwrite(segme_name, seg_image)
			
			progress_info.update_and_show( suffix = '  loss {:.4f},  acc: {:.3f}'.format(ae_val_loss, a) )
		print()
		
		summary  = sess.run(merged_summary)
		test_writer.add_summary(summary, epoch)
		
	
	train_writer.close()
	test_writer.close()
	
	#saver.save(sess, os.path.join(session_modeldir, 'model.ckpt'))

#print('\nTraining completed!\nNetwork model is saved in  {}\nTraining logs are saved in {}'.format(session_modeldir, session_logdir))

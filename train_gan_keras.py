import tensorflow as tf

import progressbar as pb
import numpy as np
import networks_keras
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


# dynamic memory allocation
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.log_device_placement = False



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

input_shape = Xh.shape[1:]

# shuffling data
np.random.seed(1994)
np.random.shuffle(Xh)
np.random.shuffle(Xu)

# take same number of positive and negative examples
min_size = 100 #min(Xh.shape[0], Xu.shape[0])
Xh = Xh[:min_size, ...]
Xu = Xu[:min_size, ...]

# data normalization
xmean = 127.5
xstd = 127.5

Xh = (Xh - xmean) / xstd
XU = (Xu - xmean) / xstd

# splitting sets in training and test data
size_ht = int(Xh.shape[0] * 0.8)		# size of healthy images training set
size_ut = int(Xu.shape[0] * 0.8)		# size of unhealthy images training set
print("healthy-unhealthy training set sizes:", size_ht, size_ut)

x_train_healthy, x_train_disease = Xh[:size_ht , ...], Xu[:size_ut , ...]
x_test_healthy,  x_test_disease  = Xh[ size_ht:, ...], Xu[ size_ut:, ...]

xh_train_dataset = tf.data.Dataset.from_tensor_slices(x_train_healthy).shuffle(size_ht).batch(BATCH_SIZE)
xu_train_dataset = tf.data.Dataset.from_tensor_slices(x_train_disease).shuffle(size_ut).batch(BATCH_SIZE)

xh_test_dataset = tf.data.Dataset.from_tensor_slices(x_test_healthy).batch(BATCH_SIZE)
xu_test_dataset = tf.data.Dataset.from_tensor_slices(x_test_disease).batch(BATCH_SIZE)




########################## NETWORKS ####################################

# image autoencoder network initialization (start from images with disease)
generator = networks_keras.generator_bipath_net(input_shape)
# classification network initialization
discriminator = networks_keras.discriminator_net(input_shape)

adversarial_model = Model()

########################## OPTIMIZERS ####################################
generator_optimizer = tf.keras.optimizers.Adam(STEPSIZE)
discriminator_optimizer = tf.keras.optimizers.Adam(STEPSIZE)


########################## LOSSES ################################
def discriminator_loss(healthy_output, tumor_output):
    healthy_loss = tf.keras.losses.mse(tf.ones_like(healthy_output), healthy_output)
    tumor_loss = tf.keras.losses.mse(tf.zeros_like(tumor_output), tumor_output)
    total_loss = healthy_loss + tumor_loss
    return total_loss
	
def discriminator_accuracy(healthy_output, tumor_output):
    healthy_loss = tf.keras.losses.mse(tf.ones_like(healthy_output), healthy_output)
    tumor_loss = tf.keras.losses.mse(tf.zeros_like(tumor_output), tumor_output)
    total_loss = healthy_loss + tumor_loss
    return total_loss

def generator_loss_classification(tumor_output):
    return tf.keras.losses.mse(tf.ones_like(tumor_output), tumor_output)

def generator_loss_similarity(input_images, generated_images):
    return tf.keras.losses.mse(input_images, generated_images)



########################## TRAINING OPTIMIZATION STEPS ################################
def train_step(healthy_images, tumor_images):

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = generator(tumor_images, training=True)

		real_output = discriminator(healthy_images, training=True)
		fake_output = discriminator(generated_images, training=True)

		gen_loss_c = generator_loss_classification(fake_output)
		gen_loss_s = generator_loss_similarity(tumor_images, generated_images)
		gen_loss   = gen_loss_c + gen_loss_s

		disc_loss = discriminator_loss(real_output, fake_output)
		disc_acc  = discriminator_accuracy(real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

	return gen_loss, disc_loss, disc_acc
	

########################## EVALUATION STEP ################################
def eval_step(tumor_images):
	
	generated_images = generator(tumor_images, training=False)

	real_output = discriminator(healthy_images, training=False)
	fake_output = discriminator(generated_images, training=False)

	gen_loss_c = generator_loss_classification(fake_output)
	gen_loss_s = generator_loss_similarity(tumor_images, generated_images)
	gen_loss   = gen_loss_c + gen_loss_s

	disc_loss = discriminator_loss(real_output, fake_output)
	disc_acc  = discriminator_accuracy(real_output, fake_output)

	return generated_images, gen_loss, disc_loss, disc_acc




NUM_BATCHES_TRAIN = math.ceil(x_train_healthy.shape[0] / BATCH_SIZE)
NUM_BATCHES_TEST  = math.ceil(x_test_healthy.shape[0]  / BATCH_SIZE)


with tf.Session(config=configuration) as sess:


	for epoch in range(EPOCHS):
		
		print("\nEPOCH %d/%d" % (epoch+1, EPOCHS))
		
		# exponential learning rate decay
		# if (epoch + 1) % 10 == 0:
			# STEPSIZE /= 2.0,
			# generator_optimizer = tf.keras.optimizers.Adam(STEPSIZE)
			# discriminator_optimizer = tf.keras.optimizers.Adam(STEPSIZE)
		
		
		#  initialize metrics and shuffles training datasets
		loss_generator = 0
		loss_discriminator = 0
		acc_discriminator = 0
		
		progress_info = pb.ProgressBar(total=NUM_BATCHES_TRAIN, prefix=' train', show=True)
		
		# Training of the network
		for nb, (healthy_images, disease_images) in enumerate(zip(xh_train_dataset, xu_train_dataset)):
			ab = nb + 1
			
			loss_generator, acc_discriminator = train_step(healthy_images, disease_images)
			
			loss_generator += gen_loss
			loss_discriminator += disc_loss
			acc_discriminator += disc_acc
			
			suffix = '  loss gen {:.4f}, loss discr {:.4f}, acc discr: {:.3f}'.format(loss_generator/ab, loss_discriminator/ab, acc_discriminator/ab)
			progress_info.update_and_show( suffix = suffix )
		print()
		
		
		
		# initialize the test dataset and set batch normalization inference
		loss_generator = 0
		loss_discriminator = 0
		acc_discriminator = 0
		
		progress_info = pb.ProgressBar(total=NUM_BATCHES_TEST, prefix='  eval', show=True)
		
		# evaluation of the network
		for nb, disease_images in enumerate(xu_test_dataset):
			ab = nb + 1
			
			generated_images, gen_loss, disc_loss, disc_acc = eval_step(disease_images)
			
			loss_generator += gen_loss
			loss_discriminator += disc_loss
			acc_discriminator += disc_acc
			
						
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
			
			suffix = '  loss gen {:.4f}, loss discr {:.4f}, acc discr: {:.3f}'.format(loss_generator/ab, loss_discriminator/ab, acc_discriminator/ab)
			progress_info.update_and_show( suffix = suffix )
		print()
		
		summary  = sess.run(merged_summary)
		test_writer.add_summary(summary, epoch)
		
	
	train_writer.close()
	test_writer.close()
	
	saver.save(sess, os.path.join("models", 'model.ckpt'))

#print('\nTraining completed!\nNetwork model is saved in  {}\nTraining logs are saved in {}'.format(session_modeldir, session_logdir))

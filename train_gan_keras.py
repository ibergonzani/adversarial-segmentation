import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

import progressbar as pb
import numpy as np
import networks_keras
import math
import time
import cv2
import os

from sklearn.metrics import accuracy_score

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

EPOCHS = 50
STEPSIZE_DISCRIMINATOR = 4e-4
STEPSIZE_GENERATOR = 1e-4
BATCH_SIZE = 4
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

Xh.astype('float64')
Xu.astype('float64')

# shuffling data
np.random.seed(1994)
np.random.shuffle(Xh)
np.random.shuffle(Xu)

# take same number of positive and negative examples
min_size = min(Xh.shape[0], Xu.shape[0])
Xh = Xh[:min_size, ...]
Xu = Xu[:min_size, ...]

# data normalization
xmean = 127.5
xstd = 127.5

Xh = (Xh - xmean) / xstd
Xu = (Xu - xmean) / xstd

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


########################## OPTIMIZERS ####################################
generator_optimizer = tf.keras.optimizers.Adam(STEPSIZE_GENERATOR)
discriminator_optimizer = tf.keras.optimizers.Adam(STEPSIZE_DISCRIMINATOR)


########################## LOSSES ################################
def discriminator_loss(healthy_output, tumor_output):
    healthy_loss = tf.keras.losses.MSE(0.9 * tf.ones_like(healthy_output), healthy_output)
    tumor_loss = tf.keras.losses.MSE(tf.zeros_like(tumor_output), tumor_output)
    total_loss = tf.reduce_mean(healthy_loss) + tf.reduce_mean(tumor_loss)
    return total_loss

def discriminator_accuracy(healthy_output, tumor_output):

	healthy_truth = tf.ones_like(healthy_output).numpy()
	healthy_output = healthy_output.numpy() > 0.5
	tumor_truth = tf.zeros_like(tumor_output).numpy()
	tumor_output = tumor_output.numpy() > 0.5

	h_acc = accuracy_score(healthy_truth, healthy_output)
	t_acc = accuracy_score(tumor_truth, tumor_output)

	total_acc = (h_acc + t_acc) / 2

	return total_acc


def generator_loss_classification(tumor_images_classification):
    return tf.reduce_mean(tf.keras.losses.MSE(tf.ones_like(tumor_images_classification), tumor_images_classification))

def generator_loss_similarity(input_images, generated_images):
    return tf.reduce_mean(tf.keras.losses.MSE(input_images, generated_images))



########################## TRAINING OPTIMIZATION STEPS ################################
def train_step(healthy_images, tumor_images):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        noise_tumor = tf.random.normal(tumor_images.shape, mean=0, stddev=0.001, dtype=tf.dtypes.float64)
        noise_healthy = tf.random.normal(healthy_images.shape, mean=0, stddev=0.001, dtype=tf.dtypes.float64)
        noise_generated = tf.random.normal(tumor_images.shape, mean=0, stddev=0.001, dtype=tf.dtypes.float32)

        generated_images = generator(tumor_images + noise_tumor, training=True)
        generated_images = generated_images + noise_generated

        healthy_images = healthy_images + noise_healthy

        h_class = discriminator(healthy_images, training=True)
        t_class = discriminator(generated_images, training=True)

        gen_loss_c = generator_loss_classification(t_class)
        gen_loss_s = generator_loss_similarity(tumor_images, generated_images)
        gen_loss   = gen_loss_c + gen_loss_s

        disc_loss = discriminator_loss(h_class, t_class)
        disc_acc  = discriminator_accuracy(h_class, t_class)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss, disc_acc
    return 0


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

		gen_loss, disc_loss, disc_acc = train_step(healthy_images, disease_images)

		loss_generator += gen_loss.numpy().item()
		loss_discriminator += disc_loss.numpy().item()
		acc_discriminator += disc_acc.item()

		suffix = '  loss gen {:.4f}, loss discr {:.4f}, acc discr: {:.3f}'.format(loss_generator/ab, loss_discriminator/ab, acc_discriminator/ab)
		progress_info.update_and_show( suffix = suffix )
	print()



	# initialize the test dataset and set batch normalization inference
	loss_generator = 0
	loss_discriminator = 0
	acc_discriminator = 0

	progress_info = pb.ProgressBar(total=NUM_BATCHES_TEST, prefix='  eval', show=True)

	# evaluation of the network
	for nb, disease_batch in enumerate(xu_test_dataset):
		ab = nb + 1

		generated_images, gen_loss, disc_loss, disc_acc = eval_step(disease_batch) # ins = disease images batch

		loss_generator += gen_loss.numpy().item()
		loss_discriminator += disc_loss.numpy().item()
		acc_discriminator += disc_acc.item()


		if (epoch + 1) % 2 == 0:
			ins = disease_batch.numpy()
			out = generated_images.numpy()

			out_dir = os.path.join("out/", str(epoch+1))
			if not os.path.exists(out_dir):
				os.makedirs(out_dir)

			ins = (ins * xstd) + xmean
			out = (out * xstd) + xmean

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

			# saver.save(sess, os.path.join("models", 'model.ckpt'), global_step=epoch+1)

		suffix = '  loss gen {:.4f}, loss discr {:.4f}, acc discr: {:.3f}'.format(loss_generator/ab, loss_discriminator/ab, acc_discriminator/ab)
		progress_info.update_and_show( suffix = suffix )
	print()

	# summary  = sess.run(merged_summary)
	# test_writer.add_summary(summary, epoch)


# train_writer.close()
# test_writer.close()

# saver.save(sess, os.path.join("models", 'model.ckpt'))

#print('\nTraining completed!\nNetwork model is saved in  {}\nTraining logs are saved in {}'.format(session_modeldir, session_logdir))

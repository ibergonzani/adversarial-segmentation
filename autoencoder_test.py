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

EPOCHS = 30
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
size_ht = int(Xh.shape[0] * 0.85)		# size of healthy images training set
size_ut = int(Xu.shape[0] * 0.85)		# size of unhealthy images training set
print("healthy-unhealthy training set sizes:", size_ht, size_ut)

x_train_healthy, x_train_disease = Xh[:size_ht , ...], Xu[:size_ut , ...]
x_test_healthy,  x_test_disease  = Xh[ size_ht:, ...], Xu[ size_ut:, ...]

xh_train_dataset = tf.data.Dataset.from_tensor_slices(x_train_healthy).shuffle(size_ht).batch(BATCH_SIZE)
xu_train_dataset = tf.data.Dataset.from_tensor_slices(x_train_disease).shuffle(size_ut).batch(BATCH_SIZE)

xh_test_dataset = tf.data.Dataset.from_tensor_slices(x_test_healthy).batch(BATCH_SIZE)
xu_test_dataset = tf.data.Dataset.from_tensor_slices(x_test_disease).batch(BATCH_SIZE)



########################## NETWORKS ####################################

# image autoencoder network initialization (start from images with disease)
generator = networks_keras.autoencoder_net(input_shape)


########################## OPTIMIZERS ####################################
optimizer_generator = tf.keras.optimizers.Adam(STEPSIZE_GENERATOR)


########################## LOSSES ################################
def generator_loss_similarity(input_images, generated_images):
    return tf.reduce_mean(tf.keras.losses.MSE(input_images, generated_images))


########################## TRAINING OPTIMIZATION STEPS ################################
def train_step(healthy_images, tumor_images):

    with tf.GradientTape() as gen_tape:
        noise_tumor = tf.random.normal(tumor_images.shape, mean=0, stddev=0.001, dtype=tf.dtypes.float64)
        noise_healthy = tf.random.normal(healthy_images.shape, mean=0, stddev=0.001, dtype=tf.dtypes.float64)

        # adding noise
        healthy_images = healthy_images + noise_healthy
        tumor_images = tumor_images + noise_tumor

        # generating tumor free (possibly) images
        generated_images_t = generator(tumor_images, training=True)
        generated_images_h = generator(healthy_images, training=True)

        # losses
        gen_loss_s = generator_loss_similarity(tumor_images, generated_images_t)
        gen_loss_c = generator_loss_similarity(healthy_images, generated_images_h)
        gen_loss   = gen_loss_s + gen_loss_c

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    optimizer_generator.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss



########################## EVALUATION STEP ################################
def eval_step(healthy_images, tumor_images):

    generated_images_t = generator(tumor_images, training=False)
    generated_images_h = generator(healthy_images, training=False)

    gen_loss_s = generator_loss_similarity(tumor_images, generated_images_t)
    gen_loss_c = generator_loss_similarity(tumor_images, generated_images_h)
    gen_loss   = gen_loss_s  + gen_loss_c


    return generated_images_t, generated_images_h, gen_loss




NUM_BATCHES_TRAIN = math.ceil(x_train_healthy.shape[0] / BATCH_SIZE)
NUM_BATCHES_TEST  = math.ceil(x_test_healthy.shape[0]  / BATCH_SIZE)


for epoch in range(EPOCHS):

    print("\nEPOCH %d/%d" % (epoch+1, EPOCHS))

    # exponential learning rate decay
    # if (epoch + 1) % 10 == 0:
    	# STEPSIZE /= 2.0,
    	# optimizer_generator = tf.keras.optimizers.Adam(STEPSIZE)
    	# optimizer_discriminator = tf.keras.optimizers.Adam(STEPSIZE)


    #  initialize metrics and shuffles training datasets
    loss_generator = 0

    progress_info = pb.ProgressBar(total=NUM_BATCHES_TRAIN, prefix=' train', show=True)

    # Training of the network
    for nb, (healthy_images, disease_images) in enumerate(zip(xh_train_dataset, xu_train_dataset)):
        ab = nb + 1

        gen_loss = train_step(healthy_images, disease_images)
        loss_generator += gen_loss.numpy().item()

        suffix = '  LG {:.4f}'.format(loss_generator/ab)
        progress_info.update_and_show( suffix = suffix )
    print()



    # initialize the test dataset and set batch normalization inference
    loss_generator = 0

    progress_info = pb.ProgressBar(total=NUM_BATCHES_TEST, prefix='  eval', show=True)

    # evaluation of the network
    for nb, (healthy_batch, disease_batch) in enumerate(zip(xh_test_dataset, xu_test_dataset)):
        ab = nb + 1
        disease_batch = disease_batch + tf.random.normal(disease_batch.shape, mean=0, stddev=0.001, dtype=tf.dtypes.float64)
        generated_images_t, generated_images_h, gen_loss = eval_step(healthy_batch, disease_batch) # ins = disease images batch

        loss_generator += gen_loss.numpy().item()

        if (epoch + 1) % 2 == 0:
            ins = tf.concat([disease_batch, healthy_batch], axis=0).numpy()
            out =  tf.concat([generated_images_t, generated_images_h], axis=0).numpy()

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

        suffix = '  LG {:.4f}'.format(loss_generator/ab)
        progress_info.update_and_show( suffix = suffix )
    print()

    # summary  = sess.run(merged_summary)
    # test_writer.add_summary(summary, epoch)


# train_writer.close()
# test_writer.close()

generator.save(os.path.join("models", 'autoencoder.h5'))

print('\nTraining completed!\nNetwork model is saved in  ./models\nTraining logs are saved in ')

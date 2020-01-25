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

EPOCHS = 100
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
Xh, Xu = load_brain_dataset(scale=0.5) # load_messidor_dataset_binary(1)
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
encoder, decoder = networks_keras.encoder_decoder_net(input_shape, 1024)
discriminator = networks_keras.discriminator_net(input_shape)
double_encoder, _ = networks_keras.encoder_decoder_net(input_shape, 1024)


########################## OPTIMIZERS ####################################
optimizer_autoencoder = tf.keras.optimizers.Adam(STEPSIZE_GENERATOR)
optimizer_decoder = tf.keras.optimizers.Adam(STEPSIZE_GENERATOR)
optimizer_discriminator = tf.keras.optimizers.Adam(STEPSIZE_DISCRIMINATOR)
optimizer_d_enc = tf.keras.optimizers.Adam(STEPSIZE_GENERATOR)

########################## LOSSES ################################

def loss_generator_similarity(input_images, generated_images):
    return tf.reduce_mean(tf.keras.losses.MAE(input_images, generated_images))

def loss_generator_classification(generated_classes):
    return tf.reduce_mean(tf.keras.losses.MSE(tf.ones_like(generated_classes), generated_classes))


def loss_discriminator_classification(real_images_classes, fake_images_classes):
    l1 = tf.reduce_mean(tf.keras.losses.MSE(0.9 * tf.ones_like(real_images_classes), real_images_classes))
    l2 = tf.reduce_mean(tf.keras.losses.MSE(0.0 * tf.ones_like(fake_images_classes), fake_images_classes)) # one sided label smoothing
    return (l1 + l2) / 2

def accuracy_discriminator_classification(real_images_classes, fake_images_classes):
    	real_classes = tf.ones_like(real_images_classes).numpy()
    	real_output = real_images_classes.numpy() > 0.5
    	fake_classes = tf.zeros_like(fake_images_classes).numpy()
    	fake_output = fake_images_classes.numpy() > 0.5

    	real_acc = accuracy_score(real_classes, real_output)
    	fake_acc = accuracy_score(fake_classes, fake_output)

    	accuracy = (real_acc + fake_acc) / 2
    	return accuracy

def loss_double_encoder(latent_real, latent_gen):
    return tf.reduce_mean(tf.keras.losses.MSE(latent_real, latent_gen))


########################## TRAINING OPTIMIZATION STEPS ################################
def train_step(healthy_images, generator_phase):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as ddenc_tape:

        # adding noise
        noise_healthy = tf.random.normal(healthy_images.shape, mean=0, stddev=0.001, dtype=tf.dtypes.float64)
        healthy_images = healthy_images + noise_healthy

        # generating healthy images
        latent_h = encoder(healthy_images, training=True)
        generated_images_h = decoder(latent_h, training=True)

        real_classification = discriminator(healthy_images, training=True)
        fake_classification = discriminator(generated_images_h, training=True)

        latent_z = double_encoder(generated_images_h, training=True)

        # losses
        loss_autoencoder = loss_generator_similarity(healthy_images, generated_images_h)
        loss_decoder = 50 * loss_generator_classification(fake_classification)

        loss_discriminator = loss_discriminator_classification(real_classification, fake_classification)
        acc_discriminator  = accuracy_discriminator_classification(real_classification, fake_classification)

        loss_d_enc = loss_double_encoder(latent_h, latent_z)

    # computing gradients of losses wrt model parameters
    gradients_of_decoder       = dec_tape.gradient(loss_decoder, decoder.trainable_variables)
    gradients_of_autoencoder   = gen_tape.gradient(loss_autoencoder, encoder.trainable_variables + decoder.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(loss_discriminator, discriminator.trainable_variables)
    gradients_of_d_enc         = ddenc_tape.gradient(loss_d_enc, encoder.trainable_variables + decoder.trainable_variables + double_encoder.trainable_variables)

    # applying gradient
    if generator_phase:
        optimizer_decoder.apply_gradients(zip(gradients_of_decoder, decoder.trainable_variables))
        optimizer_autoencoder.apply_gradients(zip(gradients_of_autoencoder, encoder.trainable_variables + decoder.trainable_variables))
        optimizer_d_enc.apply_gradients(zip(gradients_of_d_enc, encoder.trainable_variables + decoder.trainable_variables + double_encoder.trainable_variables))
    else:
        optimizer_discriminator.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return loss_autoencoder, loss_discriminator, acc_discriminator



########################## EVALUATION STEP ################################
def eval_step(healthy_images):

    latent_h = encoder(healthy_images, training=False)
    generated_images_h = decoder(latent_h, training=False)

    real_classification = discriminator(healthy_images, training=False)
    fake_classification = discriminator(generated_images_h, training=False)

    # losses
    loss_autoencoder = loss_generator_similarity(healthy_images, generated_images_h)
    loss_decoder = 50 * loss_generator_classification(fake_classification)

    loss_discriminator = loss_discriminator_classification(real_classification, fake_classification)
    acc_discriminator  = accuracy_discriminator_classification(real_classification, fake_classification)

    return loss_autoencoder, loss_discriminator, acc_discriminator




# GENERATION OF FAKE IMAGES TUMOR FREE
def generate(images):
    latent    = encoder(images, training=False)
    generated = decoder(latent, training=False)
    return generated




def save_generated_images(epoch, nb, ins, out):
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

    return





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
    loss_discriminator = 0
    acc_discriminator = 0

    progress_info = pb.ProgressBar(total=NUM_BATCHES_TRAIN, prefix=' train', show=True)

    # Training of the network
    for nb, healthy_images in enumerate(xh_train_dataset):
        ab = nb + 1

        gan_training_phase = (epoch % 2 == 1)
        gen_loss, disc_loss, acc_loss = train_step(healthy_images, gan_training_phase)

        loss_generator += gen_loss.numpy().item()
        loss_discriminator += disc_loss.numpy().item()
        acc_discriminator += acc_loss.item()

        suffix = '  LG {:.4f}, LD {:.4f}, AD: {:.3f}'.format(loss_generator/ab, loss_discriminator/ab, acc_discriminator/ab)
        progress_info.update_and_show( suffix = suffix )
    print()



    # initialize the test dataset and set batch normalization inference
    loss_generator = 0
    loss_discriminator = 0
    acc_discriminator = 0

    progress_info = pb.ProgressBar(total=NUM_BATCHES_TEST, prefix='  eval', show=True)


    # evaluation of the network
    for nb, healthy_batch in enumerate(xh_test_dataset):
        ab = nb + 1

        gen_loss, disc_loss, acc_loss = eval_step(healthy_batch)

        loss_generator += gen_loss.numpy().item()
        loss_discriminator += disc_loss.numpy().item()
        acc_discriminator += acc_loss.item()

        suffix = '  LG {:.4f}, LD {:.4f}, AD: {:.3f}'.format(loss_generator/ab, loss_discriminator/ab, acc_discriminator/ab)
        progress_info.update_and_show( suffix = suffix )


    # generating images from one with tumor
    if (epoch + 1) % 2 == 0:
        for nb, disease_batch in enumerate(xu_test_dataset):
            disease_batch = disease_batch + tf.random.normal(disease_batch.shape, mean=0, stddev=0.001, dtype=tf.dtypes.float64)

            ins = disease_batch.numpy()
            out = generate(disease_batch).numpy()
            save_generated_images(epoch, nb, ins, out)

            # saver.save(sess, os.path.join("models", 'model.ckpt'), global_step=epoch+1)
    print()

    # summary  = sess.run(merged_summary)
    # test_writer.add_summary(summary, epoch)


# train_writer.close()
# test_writer.close()

encoder.save(os.path.join("models", 'encoder.h5'))
decoder.save(os.path.join("models", 'generator.h5'))
discriminator.save(os.path.join("models", 'discriminator.h5'))

print('\nTraining completed!\nNetwork model is saved in  ./models\nTraining logs are saved in ')

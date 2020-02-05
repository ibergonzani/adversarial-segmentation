import numpy as np
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

import statistics
import cv2
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score



DATASET_IMAGE_PATH = "dataset/brain_tumor_dataset"
DATASET_MASK_PATH  = "dataset/brain_mask_dataset"
DATASET_GEN_IMAGE_PATH = "dataset/brain_gen_tumor_dataset"
DATASET_GEN_MASK_PATH  = "dataset/brain_gen_mask_dataset"
DATASET_GEN_DIFFS_PATH  = "dataset/brain_gen_diffs_dataset"

HEALTHY_DATASET_PATH = "dataset/"

MODEL_ENCODER_PATH = "encoder.h5"
MODEL_DECODER_PATH = "generator.h5"


############# PREPROCESSING FUNCTIONS ##########################################

def resize(image, scale):
	h = int(image.shape[0] * scale)
	w = int(image.shape[1] * scale)

	ns = list(image.shape)
	ns[1] = h
	ns[0] = w

	return cv2.resize(image, tuple(ns))


def scale_and_expand(image):
    image = resize(image, 0.5)
    image = image[..., np.newaxis]
    return image

def expand(image):
    image = image[..., np.newaxis]
    return image




######################## DATASET LOADING FUNCTION ##############################

def load_categorical_images(folder_path, preprocess=None):
    images = list()
    labels = list()
    names = list()

    classes = [int(x) for x in os.listdir(folder_path)]

    for c in classes:
        image_class_folder = os.path.join(DATASET_IMAGE_PATH, str(c))
        image_names = os.listdir(image_class_folder)

        for image_name in image_names:
            image_path = os.path.join(image_class_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            if not preprocess is None:
                image = preprocess(image)

            names.append(image_name)
            images.append(image)
            labels.append(c)

    return images, labels, names



def load_dataset():
    images, l, n = load_categorical_images(DATASET_IMAGE_PATH, scale_and_expand)
    masks, _, _  = load_categorical_images(DATASET_MASK_PATH, scale_and_expand)
    return images, l, masks, n



def load_generated_data():
    images, l, n = load_categorical_images(DATASET_GEN_IMAGE_PATH, expand)
    masks, _, _  = load_categorical_images(DATASET_GEN_MASK_PATH, expand)
    return images, l, masks, n



def load_healthy():
    pass



######################## IMAGES SAVE FUCNTIONS #################################

def save_categorical_images(path, images, labels, names):
    for imm, lbl, nm in zip(images, labels, names):
        folder_path = os.path.join(path, str(lbl))
        os.makedirs(folder_path, exist_ok=True)

        image_path = os.path.join(folder_path, nm)
        cv2.imwrite(image_path, imm)
    return



def save_generated_data(names, labels, gen_images, gen_masks, diffs):
    save_categorical_images(DATASET_GEN_IMAGE_PATH, gen_images, labels, names)
    save_categorical_images(DATASET_GEN_MASK_PATH, gen_masks, labels, names)
    save_categorical_images(DATASET_GEN_DIFFS_PATH, diffs, labels, names)
    return




################################################################################
########################## PROCESSING AND EVALUATION ###########################
################################################################################



def load_model():
    encoder = tf.keras.models.load_model(MODEL_ENCODER_PATH)
    decoder = tf.keras.models.load_model(MODEL_DECODER_PATH)
    return encoder, decoder


mean_val  = 127.5
hrange_val = 127.5



def process_images(images, labels, masks):

    encoder, decoder = load_model()

    total = len(images)

    generated_images = list()
    generated_masks = list()
    diff = list()

    for i, (image, mask, label) in enumerate(zip(images, masks, labels)):

        print("Processing image {:4d}/{:d}".format(i+1, total), end='\r')

        # split images in half
        upper_half = (image[:128, ...] - mean_val) / hrange_val
        lower_half = (image[128::, ...] - mean_val) / hrange_val
        lower_half = lower_half[::-1, ...]

        upper_half = upper_half[np.newaxis].astype('float64')
        lower_half = lower_half[np.newaxis].astype('float64')

        # transform them to remove tumor
        upper = decoder(encoder(upper_half)).numpy()[0]
        lower = decoder(encoder(lower_half)).numpy()[0]

        # fully reconstruct the transformed image
        output = (np.concatenate((upper, lower[::-1, ...]), 0) * hrange_val) + mean_val

        # generate tumor mask by means of a comparison with the original
        difference = np.abs(image - output)

        output_mask = np.max(difference, axis=2, keepdims=True)
        output_mask[output_mask >= 30] = 255
        output_mask[output_mask <  30] = 0

        diff.append(difference)
        generated_images.append(output)
        generated_masks.append(output_mask)

    return generated_images, generated_masks, diff




def evaluate_generated_images(original_masks, generated_masks, labels):

    # metrics vars
    metrics = [accuracy_score, precision_score, recall_score]

    results = {}
    for metric in metrics:
        results[metric.__name__] = {}
        for label in set(labels):
            results[metric.__name__][label] = list()

    results["specificity"] = {}
    for label in set(labels):
        results["specificity"][label] = list()

    total = len(generated_images)
    for i, (mask, output_mask, label) in enumerate(zip(original_masks, generated_masks, labels)):
        print("Evaluating image {:4d}/{:d}".format(i+1, total), end='\r')

        mask = (mask > 128).astype('int').flatten()
        output_mask = (output_mask > 128).astype('int').flatten()

        for metric in metrics:
            value = metric(mask, output_mask)
            results[metric.__name__][label].append(value)

        spec = recall_score(mask, output_mask, 1)
        results["specificity"][label].append(spec)


    # print evaluation results
    print("\n")
    for metric in results.keys():
        print("\n" + metric.upper())

        all_values = list()
        for label, class_vals in results[metric].items():
            all_values += class_vals
            mean_cat = statistics.mean(class_vals)
            var_cat  = statistics.variance(class_vals)
            print('Tumor ' + str(label) + ":", mean_cat, "+/-", var_cat)

        total_mean = statistics.mean(all_values)
        total_var  = statistics.variance(all_values)
        print('Global:', total_mean, "+/-", total_var)

    return results


# def fit_pca_on_healthy_masks(X, num_components=1000):
#     X = sklearn.datasets.load_iris().data
#     mu = np.mean(X, axis=0)
#
#     pca = sklearn.decomposition.PCA(num_components=num_components)
#     pca.fit(X)
#
#     projection = pca.transform()
#     reconstruction = pca.inverse_transform(projection)
#     Xhat += mu
#
#     return pca
#
# print(Xhat[0,])



if __name__ == "__main__":
    images, labels, masks, names = load_dataset()

    generated_images, generated_masks, diff = process_images(images, labels, masks)
    save_generated_data(names, labels, generated_images, generated_masks, diff)

    results = evaluate_generated_images(masks, generated_masks, labels)




#

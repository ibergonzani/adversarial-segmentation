import numpy as np
import cv2
import os

# image properties
GREYSCALE = True
RESIZE = True
IMAGE_SIZE = (128, 128)

# dataset properties
MIAS_DATASET_PATH = "dataset/mias"
MIAS_TARGET_FILE = "target.txt"

# ouptut files specifications
OUTPUT_POSITIVE_PATH = "dataset/mias/positive"
OUTPUT_NEGATIVE_PATH = "dataset/mias/negative"
OUTPUT_CONTOUR_PATH  = "dataset/mias/contour"


def get_images_in(folder, ext):
	files = [x for x in os.listdir(folder) if os.path.isfile(os.path.join(folder, x))]
	images = [x for x in files if x.endswith(ext)]
	return images
	

	
def load_mias_dataset():
	negative = list()
	positive = list()
	
	for image_name in get_images_in(OUTPUT_NEGATIVE_PATH, ".png"):
		image = cv2.imread(os.path.join(OUTPUT_NEGATIVE_PATH, image_name), cv2.IMREAD_GRAYSCALE)
		image = np.reshape(image, image.shape + (1,))
		negative.append(image)
	
	for image_name in get_images_in(OUTPUT_POSITIVE_PATH, ".png"):
		image = cv2.imread(os.path.join(OUTPUT_POSITIVE_PATH, image_name), cv2.IMREAD_GRAYSCALE)
		image = np.reshape(image, image.shape + (1,))
		positive.append(image)
	
	return np.array(negative), np.array(positive)


def preprocess_images():
	
	if not os.path.isdir(MIAS_DATASET_PATH):
		print("Mias dataset folder does not exist:", MIAS_DATASET_PATH)
		return
		
	if not os.path.isdir(OUTPUT_NEGATIVE_PATH):
		os.makedirs(OUTPUT_NEGATIVE_PATH)
	if not os.path.isdir(OUTPUT_POSITIVE_PATH):
		os.makedirs(OUTPUT_POSITIVE_PATH)
	
	with open(os.path.join(MIAS_DATASET_PATH, MIAS_TARGET_FILE)) as targets_file:
		for target_line in targets_file.readlines():
			target = target_line.replace('\n','').split()
			image_name = target[0] + ".pgm"
			diagnoses = target[2]
			
			image = cv2.imread(os.path.join(MIAS_DATASET_PATH, image_name), cv2.IMREAD_GRAYSCALE)
			imw, imh = image.shape[0], image.shape[1]
			scale_x, scale_y = 1, 1
			
			if RESIZE:
				scale_w = imw / IMAGE_SIZE[0]
				scale_h = imw / IMAGE_SIZE[1]
				image = cv2.resize(image, IMAGE_SIZE)
			
			image_name = image_name[:-3] + "png"
			
			if diagnoses == "NORM":
				cv2.imwrite(os.path.join(OUTPUT_NEGATIVE_PATH, image_name), image)
			else:
				cv2.imwrite(os.path.join(OUTPUT_POSITIVE_PATH, image_name), image)	
	return

	

if __name__ == "__main__":
	preprocess_images()
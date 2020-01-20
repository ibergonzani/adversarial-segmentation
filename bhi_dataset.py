import numpy as np
import cv2
import os

# image properties
GREYSCALE = False
SPLIT = 0.3

# dataset properties
BHI_DATASET_PATH = "dataset/breast-histopathology-images"

def get_folders_in(folder):
	folders = [x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))]
	return folders

def get_images_in(folder, ext):
	files = [x for x in os.listdir(folder) if os.path.isfile(os.path.join(folder, x))]
	images = [x for x in files if x.endswith(ext)]
	return images
	

	
def load_bhi_dataset():
	negative = list()
	positive = list()
	
	dataset_folders = get_folders_in(BHI_DATASET_PATH)
	dataset_folders.remove("IDC_regular_ps50_idx5")
	lendf = len(dataset_folders)
	
	dataset_folders = dataset_folders[:int(lendf * SPLIT)]
	
	for folder_name in dataset_folders:
	
		folder_path = os.path.join(BHI_DATASET_PATH, folder_name)
		negative_folder_path = os.path.join(folder_path, "0")
		positive_folder_path = os.path.join(folder_path, "1")
		
		for image_name in get_images_in(negative_folder_path, ".png"):
			image = cv2.imread(os.path.join(negative_folder_path, image_name), cv2.IMREAD_COLOR)
			if image.size == 0 or image.size != 7500:
				print("error", positive_folder_path, image_name)
				continue
			#image = np.reshape(image, image.shape + (1,))
			negative.append(image)
		
		for image_name in get_images_in(positive_folder_path, ".png"):
			image = cv2.imread(os.path.join(positive_folder_path, image_name), cv2.IMREAD_COLOR)
			if image.size == 0 or image.size != 7500:
				print("error", positive_folder_path, image_name)
				continue
			#image = np.reshape(image, image.shape + (1,))
			positive.append(image)
			
	return np.array(negative), np.array(positive)
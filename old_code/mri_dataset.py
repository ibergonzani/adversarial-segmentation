from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import glob
import cv2
import os


# payload = {"from_select":"1", "modality_value":"T2", "slice_value":"3mm",
			# "noise_value":"pn3", "field_value":"rf20","download":"[Download]"}	

# sess = requests.Session()

# r = sess.post("https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1", data = payload)
# print(r.text)

STARE_CODES = "dataset/STARE/stare_codes.txt"
STARE_DIAGNOSES = "dataset/STARE/stare_diagnoses.txt"
STARE_IMAGES_FOLDER = "dataset/STARE/images/"

def load_stare_dataset():
	
	CLASSES = 0
	
	codes = list()
	images = list()
	normal_ids = list()
	images_ids = list()
	
	with open(STARE_CODES) as fc:
		for line in fc.readlines():
			token = line.replace('\n','').split('\t')
			codes.append(token[1:])
	
	CLASSES = len(codes)
	print("Classes:", CLASSES, "\n", codes)
	
	for image_path in glob.glob(STARE_IMAGES_FOLDER + "*.ppm"):
		print("READ", image_path)
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		print("Image shape:", image.shape)
		image = cv2.resize(image, (128,128))
		images.append(np.reshape(image, image.shape + (1,)))
		
		image_id = int(image_path[-7:-4])
		images_ids.append(image_id)
		
	n_images = len(images)
	print("Images count:", n_images)
	
	diagnoses = np.zeros((n_images, CLASSES))
	
	with open(STARE_DIAGNOSES) as fd:
		print("READING DIAGNOSES")
		count = 0
		for line in fd.readlines():
			diagnose_id = int(line[3:6])
			if not diagnose_id in images_ids:
				continue
				
			sample_codes = [int(s) for s in line.split() if s.isdigit()]
			
			if 0 in sample_codes:
				diagnoses[count, 0] = 1
				normal_ids.append(count)
			else:
				for category in sample_codes:
					diagnoses[count, category] = 1
			
			count += 1
				

	print("Class occurencies:")
	print(np.sum(diagnoses, axis=0), normal_ids)
	
	X = np.array(images)
	Y = diagnoses
	
	print("DIAGNOSES: ", X.shape, Y[normal_ids,:].shape)
	
	return X, Y, normal_ids


	
def load_stare_dataset_separated():
	X, Y, normal_ids = load_stare_dataset()
	
	ids_range = range(X.shape[0])
	other_ids = [x for x in ids_range if not x in normal_ids]
	
	return X[normal_ids,...], X[other_ids,...]	# not returning labels 0 normal 1 not normal
	
	



	

MESSIDOR_PATH = 'dataset/messidor/'

def load_messidor_dataset(sets=list(), transform=lambda x:x):
	
	# getting desired parts of dataset
	folders = os.listdir(MESSIDOR_PATH)
	folders = [os.path.join(MESSIDOR_PATH, x) for x in folders if int(x) in sets or not sets]
	folders = [x for x in folders if os.path.isdir(x)]
	
	X = list()
	Y = list()
	
	# reading data (input-output)
	for folder in folders:
	
		for file in glob.glob(os.path.join(folder, '*.xls')):
		
			# getting data description (paths, diagnoses)
			data = pd.read_excel(file)
			Y.extend(data.iloc[:,[2,3]].values)
			
			# getting images	
			for image_name in data.iloc[:,0]: # image name
				image = cv2.imread(os.path.join(folder, image_name))
				image = transform(image)
				cv2.imwrite("out/_" + str(Y[-1][0]) + "__" + str(image_name) + ".png", image)
				X.append(image)
	
	Y = np.array(Y)
	X = np.array(X)
	
	return X, Y
	
# load images from messidor dataset in 2 different arrays
# one array
def load_messidor_dataset_binary(output, sets=list(), transform=lambda x:x):
	X, Y = load_messidor_dataset(sets, transform);
	
	Y = Y[:, output]
	
	idx_healty   = np.where(Y == 0)[0]
	idx_unhealty = np.where(Y != 0)[0]
	
	return X[idx_healty, ...], X[idx_unhealty, ...]
	
	
# resize all the images contained in data to the size given in shape
# data must be of the form (batch, width, height, channel)
# shape contains new width and height
def resize_dataset_images(data, shape):
	resize_data = np.array((data.shape[0], shape[0], shape[1], data.shape[3]))
	for n in range(data.shape[0]):
		resize_data[n,...] = cv2.resize(data[n,...], shape)
	return resize_data
	


# transformaztion applied to the images of the messidor dataset
def messidor_transform(image):
	image = image[:, 400:1820, :] 					# crop image to remove black stripes
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	# convert to grayscale
	image = cv2.resize(image, (128, 128))			# reduce dimension
	image = np.reshape(image, image.shape + (1,))	# add channels dimension
	return image
	

	
	
# testing functions
if __name__ == "__main__":
	Xh, Xu = load_messidor_dataset_binary(1, transform=messidor_transform)	
	Xh.save("messidor_healty")
	Xu.save("messidor_unhealty")
	# X, Y = load_stare_dataset()
	# X, Y = load_messidor_dataset([1,3])
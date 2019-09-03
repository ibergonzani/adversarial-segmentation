from bs4 import BeautifulSoup
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
	diagnoses = list()
	
	with open(STARE_CODES) as fc:
		for line in fc.readlines():
			token = line.replace('\n','').split('\t')
			codes.append(token[1:])
	
	CLASSES = len(codes)
	print("Classes:", CLASSES, "\n", codes)
	
	for image_path in glob.glob(STARE_IMAGES_FOLDER + "*.ppm"):
		print("READ", image_path)
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		images.append(image)
	
	
	with open(STARE_DIAGNOSES) as fd:
		print("READING DIAGNOSES")
		for line in fd.readlines():
			token = line.split('\t')
			diagnose = np.zeros((CLASSES, 1))
			sample_codes = [int(s) for s in line.split() if s.isdigit()]
			print(sample_codes)
			for category in sample_codes:
				diagnose[category, 0] = 1
			diagnoses.append(diagnose)
			print(diagnose)

	
	X = np.array(images)
	Y = np.array(diagnoses)
	
	print("DIAGNOSES: ", X.shape, Y.shape)
	
	return X, Y

	
	
if __name__ == "__main__":
	load_stare_dataset()
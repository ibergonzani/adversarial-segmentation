import numpy as np
import cv2
import os


def load_dataset(folder):

	X = list()
	Y = list()
	
	for file in os.listdir(folder):

		image = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
		
		label = 0 if file[-5:] == 'h.png' else 1
		
		X.append(np.reshape(image, image.shape + (1,)))
		Y.append([label])
		
	return np.array(X, dtype=np.float32), np.array(Y)
	
	
	
def load_dataset_separated(folder):

	X, Y = load_dataset(folder)
	
	index_healty   = np.where(Y == 0)[0]
	index_unhealty = np.where(Y == 1)[0]
	
	return X[index_healty,...], X[index_unhealty,...]

	
	


if __name__ == '__main__':
	
	N_IMAGES = 200
	HEIGHT = 256
	WIDTH = 256

	FILES_NAME = 'image_'
	OUTPUT_FOLDER = 'phantom/train/'

	if not os.path.exists(OUTPUT_FOLDER):
		os.mkdir(OUTPUT_FOLDER)


	print("Phantom Dataset creation started");

	for i in range(2*N_IMAGES):
		image =  np.zeros((HEIGHT,WIDTH,3), np.float32)
		
		# greyscale colors for the different shapes
		torso_color = 3 * (np.random.randint(50, 60),)
		lung_color  = 3 * (np.random.randint(70, 80),)
		esof_color  = 3 * (np.random.randint(70, 80),)
		stom_color  = 3 * (np.random.randint(90, 110),)
		ribs_color  = 3 * (np.random.randint(100, 120),)
		
		# draw basic shape (torso)
		torso_w = np.random.randint(90, 105)
		torso_h = np.random.randint(130, 140)
		torso_c = (128, 128)
		cv2.ellipse(image, torso_c, (torso_w,torso_h), 0, 0, 360, torso_color, -1)
		
		# draw lungs
		lung_w = np.random.randint(34, 40, 2)
		lung_h = np.random.randint(65, 70, 2)
		lung_c = ((82,100), (175,100))
		cv2.ellipse(image, lung_c[0], (lung_w[0],lung_h[0]), 0, 0, 360, lung_color, -1)
		cv2.ellipse(image, lung_c[1], (lung_w[1],lung_h[1]), 0, 0, 360, lung_color, -1)
		
		# draw esofagus
		esof_w = np.random.randint(5, 8)
		esof_h = np.random.randint(85, 95)
		esof_c = (128,90)
		cv2.ellipse(image, esof_c, (esof_w,esof_h), 0, 0, 360, esof_color, -1)
		
		# draw stomach
		stom_w = np.random.randint(70, 80)
		stom_h = np.random.randint(20, 25)
		stom_c = (128,200)
		cv2.ellipse(image, stom_c, (stom_w,stom_h), 0, 0, 360, stom_color, -1)
		
		
		# draw ribs
		for rib in range(8):
			ribs_color  = 3 * (np.random.randint(100, 120),)
		
			side = rib % 2
			pose = rib // 2
			
			rib_w = np.random.randint(32+3*pose, 35+3*pose)
			rib_h = np.random.randint(4, 6)
			rib_c = (128 - (rib_w+5) + 2*side*(rib_w+5), 50 + 20*pose)
			
			cv2.ellipse(image, rib_c, (rib_w,rib_h), 0, 0, 360, ribs_color, -1)
		
		# add noise
		noise = np.random.randint(0, 10, (HEIGHT,WIDTH))
		image[:,:,0] = image[:,:,0] + noise
		image[:,:,1] = image[:,:,1] + noise
		image[:,:,2] = image[:,:,2] + noise
		
		STR_MOD = 'h' if i < N_IMAGES else 'u'
		
		if i >= N_IMAGES:
		
			# add extraneous corps
			for n in range(np.random.randint(25)):
				corp_r = np.random.randint(1, 4)
				corp_side = (np.random.randint(1,3) < 2)
				corp_x = np.random.randint(lung_c[corp_side][0] - lung_w[side], lung_c[corp_side][0] + lung_w[side])
				corp_y = np.random.randint(lung_c[corp_side][1] - lung_h[side], lung_c[corp_side][1] + lung_h[side])
				
				corp_color = 3 * (np.random.randint(60, 70),);
				
				layer = np.zeros((WIDTH, HEIGHT, 3), np.float32)
				cv2.ellipse(layer, (corp_x,corp_y), (corp_r,corp_r), 0, 0, 360, corp_color, -1)
				
				image = image + layer
				
		
		# limit to max 256
		image = np.clip(image, 0, 255)
		image = cv2.resize(image, (128,128))
		
		image_path = OUTPUT_FOLDER + FILES_NAME + '{:04d}'.format(i) + STR_MOD + '.png';
		cv2.imwrite(image_path, image);
		
		print("Phantom image created " + str(i+1) + "/" + str(2*N_IMAGES), end='\r');

	print("\nPhantom Dataset creation completed");
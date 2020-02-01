from bs4 import BeautifulSoup
import requests
import time
import csv
import os

MMSET_URL = "http://peipa.essex.ac.uk/info/mias.html"
MMSET_IMAGES_URL = "http://peipa.essex.ac.uk/pix/mias/"

def request_webpage(url, max_requests=15, sleep_time=0.0, error_sleep_time=5.0):
	time.sleep(sleep_time)
	completed = False
	webpage = None
	ntry = 0
	
	while not completed and ntry < max_requests:
		ntry += 1
		try:
			r = requests.get(url)
			if r.status_code == 200:
				webpage = BeautifulSoup(r.text, 'html.parser')
				completed = True
		except:
			print("Connection error. Waiting {:f} sec.".format(error_sleep_time))
			time.sleep(error_sleep_time)
	return webpage
	

def download_dataset(folder):
	data_webpage = request_webpage(MMSET_URL)
	images_webpage = request_webpage(MMSET_IMAGES_URL)
	
	# extracting data informations
	box = next_page.find("ul", {"id": "search-result"})
	
	ads = list()
	ads += box.findAll("li", {"class": "item topad result gtm-search-result"})
	
	# extracting and downloading images informations
	box = next_page.find("ul", {"id": "search-result"})
	
	images_name = list()
	images_name += box.findAll("li", {"class": "item topad result gtm-search-result"})


 cv2.imread('a.pgm',-1)
	
	
	
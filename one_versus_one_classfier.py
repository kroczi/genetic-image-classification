#! /usr/bin/python3
import os

import csv
import numpy as np
import png

from area import Histogram
from image import Image
from data_types import Floats, Floats2, Floats3, Shape, Position, Size, Index, \
					   HoG, bins1, bins2, bins3, distance1, distance2, distance3

CLASSES = 41
CLASSIFIERS_FILE = "../../Dane/image_classification_datasets/motion_tracking/motion_tracking_classifiers.txt"
IMAGE_FILE = "../../Dane/image_classification_datasets/motion_tracking/training/30/0021.png"

BASE_DIRECTORY = '../../Dane/image_classification_datasets/motion_tracking/'

context = {
	"__builtins__" : None,
	"add": Floats.add, "sub": Floats.sub, "mul": Floats.mul, "div": Floats.div,
	"add2": Floats2.add2, "sub2": Floats2.sub2, "mul2": Floats2.mul2, "div2": Floats2.div2,
	"add3": Floats3.add3, "sub3": Floats3.sub3, "mul3": Floats3.mul3, "div3": Floats3.div3,
	"bins1": bins1, "bins2": bins2, "bins3": bins3,
	"distance1": distance1, "distance2": distance2, "distance3": distance3,
	"HoG": HoG,
	"Image": Image, "Shape": Shape, "Position": Position, "Size": Size, "Histogram": Histogram, "Index": Index
}

def ovo_classifier(classifiers, indices, image):
	ranking = np.zeros((CLASSES,))
	index = 0

	for i in range(CLASSES):
		for j in range(i + 1, CLASSES):
			if i in indices and j in indices:
				if classifiers[index](image) > 0:
					ranking[j] += 1
				else:
					ranking[i] += 1
			index += 1

	return ranking

with open(CLASSIFIERS_FILE) as classifiers:
	classifiers = [eval('lambda IN0: ' + x.strip(), context) for x in classifiers.readlines()]

indices = list(range(CLASSES))
image = Image(IMAGE_FILE)
ranking = ovo_classifier(classifiers, indices, image)
print(str(IMAGE_FILE) + ', ' + str(ranking.argmax()))


# indices = list(range(CLASSES))
# for c in range(CLASSES):
# 	for directory in os.listdir(BASE_DIRECTORY):
# 		if os.path.isdir(os.path.join(BASE_DIRECTORY, directory)):
# 			for filename in os.listdir(os.path.join(BASE_DIRECTORY, directory, str(c))):
# 				path = os.path.join(BASE_DIRECTORY, directory, str(c), filename)
# 				image = Image(path, c)
# 				ranking = ovo_classifier(classifiers, indices, image)
# 				print(str(path) + ', ' + str(c) + ', ' + str(ranking.argmax()))

#
#
# indices = np.where(ranking > len(indices) / 2)[0].tolist()
# print(indices)
# ranking = ovo_classifier(classifiers, indices)
#
#
# indices = np.where(ranking > len(indices) / 2)[0].tolist()
# print(indices)
# ranking = ovo_classifier(classifiers, indices)

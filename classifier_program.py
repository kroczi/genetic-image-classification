#! /usr/bin/python3
from image_classification import *
import numpy as np
import os
import png

class Classificator():

	context = {
		"__builtins__" : None,
		"add": Floats.add, "sub": Floats.sub, "mul": Floats.mul, "div": Floats.div,
		"add2": Floats2.add2, "sub2": Floats2.sub2, "mul2": Floats2.mul2, "div2": Floats2.div2,
		"add3": Floats3.add3, "sub3": Floats3.sub3, "mul3": Floats3.mul3, "div3": Floats3.div3,
		"bins": bins, "bins1": bins1, "bins2": bins2, "bins3": bins3,
		"distance": distance, "distance1": distance1, "distance2": distance2, "distance3": distance3,
		"HoG":HoG,
		"Image":Image, "Shape":Shape, "Position":Position, "Size":Size, "Histogram":Histogram, "Index":Index
	}

	def __init__(self, code, debug=False):
		evauluated_expression = 'lambda IN0: ' + code
		self.classificator = eval(evauluated_expression, self.context)
		self.debug = debug

	def lower(self, a):
		return a < 0

	def greater(self, a):
		return a > 0

	def count_correctly_classified(self, images_dir, comparison_operator):
		image_counter = 0
		correctly_classified_image_counter = 0

		if self.debug:
			print('Load {} test images for class'.format(len(os.listdir(images_dir))))
		for filename in os.listdir(images_dir):
			path_to_image = os.path.join(images_dir, filename)
			image = Image(path_to_image)
			if self.debug:
				print(str(filename) + ',' + str(self.classificator(image)))
			image_counter += 1
			if comparison_operator(self.classificator(image)) > 0:
				correctly_classified_image_counter += 1

		return (correctly_classified_image_counter, image_counter)

	def classify_positise_class(self, dir_with_images_from_class):
		image_counter = 0
		correctly_classified_image_counter = 0

		(correctly_classified_image_counter, image_counter) = self.count_correctly_classified(dir_with_images_from_class, self.greater)

		return correctly_classified_image_counter/image_counter

	def classify_negative_class(self, dir_with_images_from_class):
		image_counter = 0
		correctly_classified_image_counter = 0

		(correctly_classified_image_counter, image_counter) = self.count_correctly_classified(dir_with_images_from_class, self.lower)

		return correctly_classified_image_counter / image_counter

	def classify_pair_of_class(self, dir_with_images_from_negative_class, dir_with_images_from_positive_class):
		positive_image_counter = 0
		negative_image_counter = 0
		correctly_classified_positive_image_counter = 0
		correctly_classified_negative_image_counter = 0

		(correctly_classified_negative_image_counter, negative_image_counter) = self.count_correctly_classified(dir_with_images_from_negative_class, self.lower)
		(correctly_classified_positive_image_counter, positive_image_counter) = self.count_correctly_classified(dir_with_images_from_positive_class, self.greater)

		return (correctly_classified_negative_image_counter / negative_image_counter,
				correctly_classified_positive_image_counter / positive_image_counter,
				(correctly_classified_negative_image_counter + correctly_classified_positive_image_counter) / (positive_image_counter + negative_image_counter)
			)

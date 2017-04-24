#! /usr/bin/python3
import os

import png

from image import Image


input_path_base = '../../Dane/image_classification_datasets/coil_20_proc'
output_path_base = input_path_base + '_resized'
categories = ['training', 'testing']

for clas in range(20):
	for cat in categories:
		input_directory = os.path.join(input_path_base, cat, str(clas))
		output_directory = os.path.join(output_path_base, cat, str(clas))
		if not os.path.exists(output_directory):
			os.makedirs(output_directory)

		for filename in os.listdir(input_directory):
			image = Image(os.path.join(input_directory, filename))
			image.resize(28, 28)

			file = open(os.path.join(output_directory, filename), 'wb')
			writer = png.Writer(width=image.width, height=image.height, greyscale=True)
			writer.write(file, image.array)
			file.close()

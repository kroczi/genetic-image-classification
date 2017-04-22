#! /usr/bin/python3
from image_classification import *
import numpy as np
import os
import png

testing_images_dir = '../DataSets/coil_20_proc/testing/'
negative_class_subdir = '0'
positive_class_subdir = '1'

# Put your code, commpressed into string expression
code = 'sub(bins1(HoG(IN0, Shape(0), Position(128, 33), Size(102, 61)), Index(5)), mul2(div3(bins3(HoG(IN0, Shape(0), Position(18, 53), Size(120, 12)), Index(2)), distance3(HoG(IN0, Shape(1), Position(88, 106), Size(114, 93)), HoG(IN0, Shape(0), Position(83, 54), Size(41, 26)))), distance2(HoG(IN0, Shape(0), Position(65, 93), Size(108, 60)), HoG(IN0, Shape(0), Position(57, 71), Size(54, 85)))))'

def verify_images(classificator, images_dir):
	print('New class:')
	for filename in os.listdir(images_dir):
		path_to_image = os.path.join(images_dir, filename)
		image = Image(path_to_image)
		print(str(filename) + ',' + str(classificator(image)))

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

evauluated_expression = 'lambda IN0: ' + code
classificator = eval(evauluated_expression, context)

negative_images_dir = os.path.join(testing_images_dir, negative_class_subdir)
positivie_images_dir = os.path.join(testing_images_dir, positive_class_subdir)

verify_images(classificator, negative_images_dir)
verify_images(classificator, positivie_images_dir)

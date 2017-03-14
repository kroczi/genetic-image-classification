#! /usr/bin/python3
from image_classification import *
import numpy as np
import os
import png

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

# Put your code, commpressed into string expression
code = 'sub(bins3(HoG(IN0, Shape(1), Position(61, 67), Size(21, 106)), Index(4)), mul3(distance3(HoG(IN0, Shape(0), Position(11, 50), Size(77, 118)), HoG(IN0, Shape(0), Position(25, 52), Size(65, 37))), distance3(HoG(IN0, Shape(0), Position(116, 81), Size(72, 90)), HoG(IN0, Shape(0), Position(69, 12), Size(103, 92)))))'
evauluated_expression = 'lambda IN0: ' + code
classificator = eval(evauluated_expression, context)

images_dir = '../Dane/COIL20/'
for filename in os.listdir(images_dir):
    path_to_image = images_dir + filename 
    image = Image(path_to_image)
    print(str(filename) + ',' + str(classificator(image)))




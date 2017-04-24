#! /usr/bin/python3
import math

import numpy as np
import png

DEBUG = False

class Gradient:
	def __init__(self, lower_bin, upper_bin, lower_weight, upper_weight):
		self.lower_bin = int(lower_bin)
		self.upper_bin = int(upper_bin)
		self.lower_weight = lower_weight
		self.upper_weight = upper_weight


class Image:
	def __init__(self, path, species=None):
		if DEBUG:
			print(path)
		reader = png.Reader(path)
		(w, h, p, m) = reader.read()
		self.width = w
		self.height = h
		self.array = np.array(list(p), np.int16)
		# The class of the image
		self.species = species
		self.gradients = [[self.calculate_gradient(x, y) for x in range(self.width)] for y in range(self.height)]

	def value(self, x, y):
		return self.array[y][x]

	def calculate_gradient(self, x, y):
		if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
			return Gradient(0.0, 0.0, 0.0, 0.0)

		gx = self.value(x+1, y) - self.value(x-1, y)
		gy = self.value(x, y+1) - self.value(x, y-1)
		magnitude = math.sqrt(gx**2 + gy**2)

		orientation = math.atan2(gy, gx) * 180.0 / math.pi
		if orientation < 0:
			orientation += 360.0

		lower_bin = int(orientation // 45.0) % 8
		upper_bin = int(lower_bin + 1)

		lower_weight = magnitude * (45.0 - (orientation - lower_bin * 45.0)) / 45.0
		upper_weight = magnitude * (45.0 - (upper_bin * 45.0 - orientation)) / 45.0

		return Gradient(lower_bin, upper_bin % 8, lower_weight, upper_weight)

	def resize(self, new_width, new_height):
		tile_width = self.width // new_width
		tile_height = self.height // new_height
		new_array = np.zeros((new_width, new_height), np.float16)

		for j in range(0, new_height - 1):
			for i in range(0, new_width - 1):
				new_value = 0
				for y in range(j * tile_height, (j + 1) * tile_height - 1):
					for x in range(i * tile_width, (i + 1) * tile_width - 1):
						#print(i, j, x, y, self.width, self.height)
						new_value += self.value(x, y)
				new_array[j][i] = new_value / (tile_width * tile_height)

		i = new_width - 1
		temp_tile_width = self.width - (i * tile_width)
		for j in range(new_height - 1):
			new_value = 0
			for y in range(j * tile_height, (j + 1) * tile_height - 1):
				for x in range(i * tile_width, self.width):
					new_value += self.value(x, y)
			new_array[j][i] = new_value / (temp_tile_width * tile_height)

		j = new_height - 1
		temp_tile_height = self.height - (j * tile_height)
		for i in range(new_width - 1):
			new_value = 0
			for y in range(j * tile_height, self.height):
				for x in range(i * tile_width, (i + 1) * tile_width - 1):
					new_value += self.value(x, y)
			new_array[j][i] = new_value / (tile_width * temp_tile_height)

		i = new_width - 1
		j = new_height - 1
		new_value = 0
		for y in range(j * tile_height, self.height):
			for x in range(i * tile_width, self.width):
				new_value += self.value(x, y)
		new_array[j][i] = new_value / (temp_tile_width * temp_tile_height)

		self.width = new_width
		self.height = new_height
		self.array = new_array

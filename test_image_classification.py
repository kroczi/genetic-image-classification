#! /usr/bin/python3
import image_classification as ic
import math
import unittest
import png
import numpy as np
import os


class TestStringMethods(unittest.TestCase):

    def test_histogram_add_first(self):
        histogram = ic.Histogram()
        histogram.add(0, 4.32)
        self.assertEqual(histogram.get(0), 4.32)

    def test_histogram_add_last(self):
        histogram = ic.Histogram()
        histogram.add(7, -1.0)
        self.assertEqual(histogram.get(7), -1.0)

    def test_histogram_add_middle(self):
        histogram = ic.Histogram()
        histogram.add(3, 9.5)
        self.assertEqual(histogram.get(3), 9.5)

    def test_histogram_normalize_last_elem(self):
        histogram = ic.Histogram()
        histogram.add(0,0.0)
        histogram.add(1,1.0)
        histogram.add(2,2.0)
        histogram.add(3,3.0)
        histogram.add(4,4.0)
        histogram.add(5,5.0)
        histogram.add(6,6.0)
        histogram.add(7,7.0)
        histogram.normalize()
        
        self.assertAlmostEqual(histogram.get(7), 0.25, delta=0.01)

    def test_histogram_normalize_first_elem(self):
        histogram = ic.Histogram()
        histogram.add(0,0.0)
        histogram.add(1,1.0)
        histogram.add(2,2.0)
        histogram.add(3,3.0)
        histogram.add(4,4.0)
        histogram.add(5,5.0)
        histogram.add(6,6.0)
        histogram.add(7,7.0)
        histogram.normalize()
        
        self.assertAlmostEqual(histogram.get(0), 0.0, delta=0.01)

    def test_histogram_normalize_middle_elem(self):
        histogram = ic.Histogram()
        histogram.add(0,0.0)
        histogram.add(1,1.0)
        histogram.add(2,2.0)
        histogram.add(3,3.0)
        histogram.add(4,4.0)
        histogram.add(5,5.0)
        histogram.add(6,6.0)
        histogram.add(7,7.0)
        histogram.normalize()
        
        self.assertAlmostEqual(histogram.get(4), 0.1429, delta=0.01)

    def test_histogram_normalization_of_zeros(self):
        histogram = ic.Histogram()
        
        histogram.normalize()
        
        self.assertAlmostEqual(histogram.get(0), 0.0, delta=0.01)

    def test_distance_function_same_bin(self):
        histogram_a = ic.Histogram()
        histogram_b = ic.Histogram()

        histogram_a.add(0, 5)
        histogram_b.add(0, -5)

        self.assertEqual(ic.distance(histogram_a, histogram_b), 10)


    def test_distance_function_different_bin(self):
        histogram_a = ic.Histogram()
        histogram_b = ic.Histogram()

        histogram_a.add(0, 10)
        histogram_b.add(7, 5)

        self.assertAlmostEqual(ic.distance(histogram_a, histogram_b), math.sqrt(125), delta=0.01)

    def test_calculate_gradient_equals_zero_rectangle_area(self):
        pixels = (np.random.rand(150 * 150) < 0.5).astype(np.int16)
        pixels = pixels.reshape(150, 150)
        for x in range(0, 150):
            for y in range(0, 150):
                pixels[x][y] = 0
        pixels[0][0] = 0
        pixels[0][1] = 0
        pixels[0][2] = 0

        pixels[1][0] = 0
        pixels[1][1] = 0
        pixels[1][2] = 0

        pixels[2][0] = 0
        pixels[2][1] = 0
        pixels[2][2] = 0

        png.from_array(pixels, 'L').save("../Dane/rectangle__foo.png")
        myImage = ic.Image("../Dane/rectangle__foo.png", 0)
        area = ic.Area(myImage)
        position = ic.Position(1, 1)
        size = ic.Size(10, 10)
        rArea = ic.RectangleArea(myImage, position.x, position.y, size.height, size.width)
        histogram = rArea.create_histogram()

        self.assertAlmostEqual(histogram.get(0), 0.0, delta=0.01)
        self.assertAlmostEqual(histogram.get(1), 0.0, delta=0.01)
        os.remove("../Dane/rectangle__foo.png")

    def test_calculate_gradient_equals_zero_circle_area(self):
        pixels = (np.random.rand(150 * 150) < 0.5).astype(np.int16)
        pixels = pixels.reshape(150, 150)

        for x in range(0, 150):
            for y in range(0, 150):
                pixels[x][y] = 0
        pixels[0][0] = 0
        pixels[0][1] = 0
        pixels[0][2] = 0

        pixels[1][0] = 0
        pixels[1][1] = 0
        pixels[1][2] = 0

        pixels[2][0] = 0
        pixels[2][1] = 0
        pixels[2][2] = 0

        pixels[3][0] = 0
        pixels[3][1] = 0
        pixels[3][2] = 0

        pixels[4][0] = 0
        pixels[4][1] = 0
        pixels[4][2] = 0
        png.from_array(pixels, 'L').save("../Dane/circle__foo.png")
        myImage = ic.Image("../Dane/circle__foo.png", 0)
        area = ic.Area(myImage)
        position = ic.Position(50, 50)
        radius = 50
        cArea = ic.CircleArea(myImage, position.x, position.y, radius)
        histogram = cArea.create_histogram()

        self.assertAlmostEqual(histogram.get(0), 0.0, delta=0.01)
        self.assertAlmostEqual(histogram.get(1), 0.0, delta=0.01)
        os.remove("../Dane/circle__foo.png")

    def test_calculate_gradient_with_same_difference_rectangle_area(self):

        pixels = (np.random.rand(150 * 150) < 0.5).astype(np.int16)
        pixels = pixels.reshape(150, 150)

        for x in range(0, 150):
            for y in range(0, 150):
                pixels[x][y] = 0

        pixels[0][0] = 0
        pixels[0][1] = 100
        pixels[0][2] = 0

        pixels[1][0] = 150
        pixels[1][1] = 0
        pixels[1][2] = 250

        pixels[2][0] = 0
        pixels[2][1] = 200
        pixels[2][2] = 0

        png.from_array(pixels, 'L').save("../Dane/rectangle__foo.png")
        myImage = ic.Image("../Dane/rectangle__foo.png", 0)
        area = ic.Area(myImage)
        position = ic.Position(1, 1)
        size = ic.Size(10, 10)
        rArea = ic.RectangleArea(myImage, position.x, position.y, size.height, size.width)
        histogram=rArea.create_histogram()
        for x in range(0, 8):
            if histogram.get(x) != 0:
                # print(str(x)+": "+str(histogram.get(x)))
                continue

        self.assertAlmostEqual(histogram.get(5), 0.8600, delta=0.05)
        self.assertAlmostEqual(histogram.get(6), 0.141421, delta=0.05)
        os.remove("../Dane/rectangle__foo.png")


    def test_calculate_gradient_with_same_difference_circle_area(self):
        pixels = (np.random.rand(150 * 150) < 0.5).astype(np.int16)
        pixels = pixels.reshape(150, 150)

        for x in range(0, 150):
            for y in range(0, 150):
                pixels[x][y] = 0

        pixels[0][0] = 0
        pixels[0][1] = 100
        pixels[0][2] = 0

        pixels[1][0] = 150
        pixels[1][1] = 0
        pixels[1][2] = 250

        pixels[2][0] = 0
        pixels[2][1] = 200
        pixels[2][2] = 0

        png.from_array(pixels, 'L').save("../Dane/circle__foo.png")
        myImage = ic.Image("../Dane/circle__foo.png", 0)
        area = ic.Area(myImage)
        position = ic.Position(50, 50)
        radius = 50
        cArea = ic.CircleArea(myImage, position.x, position.y, radius)
        histogram = cArea.create_histogram()
        for x in range(0, 8):
            if histogram.get(x) != 0:
                print(str(x)+": "+str(histogram.get(x)))

        self.assertAlmostEqual(histogram.get(5), 0.0, delta=0.05)
        self.assertAlmostEqual(histogram.get(6), 0.0, delta=0.05)
        os.remove("../Dane/circle__foo.png")


if __name__ == '__main__':
    unittest.main()
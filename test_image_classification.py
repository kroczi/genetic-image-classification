#! /usr/bin/python3
import image_classification as ic
import math
import unittest



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

if __name__ == '__main__':
    unittest.main()
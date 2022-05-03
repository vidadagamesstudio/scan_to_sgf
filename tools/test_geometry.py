"""
Module to test geometry module
"""

import unittest

from tools.geometry_tools import (
    compare_line_horizontaly,
    compare_line_verticaly,
    measure_distance,
)

class TestCompareLineHorizontaly(unittest.TestCase):
    """
    class to test function compare_line_horizontaly
    """

    def test_compare_line_horizontaly_fist_before_second(self):
        """
        if the first point is before the second one the function return -1
        """
        point1 = ((1,1),(1,1))
        point2 = ((2,2),(2,2))
        comparaison = compare_line_horizontaly(point1, point2)
        self.assertEqual(comparaison, -1)

    def test_compare_line_horizontaly_fist_after_second(self):
        """
        if the first point is after the second one the function return 1
        """
        point1 = ((2,2),(2,2))
        point2 = ((1,1),(1,1))
        comparaison = compare_line_horizontaly(point1, point2)
        self.assertEqual(comparaison, 1)

    def test_compare_line_horizontaly_fist_equal_second(self):
        """
        if the first point is on the same horizontal as the second one the function return 0
        """
        point1 = ((1,1),(1,1))
        point2 = ((1,1),(1,1))
        comparaison = compare_line_horizontaly(point1, point2)
        self.assertEqual(comparaison, 0)

class TestCompareLineVertical(unittest.TestCase):
    """
    class to test function compare_line_verticaly
    """

    def test_compare_line_verticaly_fist_before_second(self):
        """
        if the first point is before the second one the function return -1
        """
        point1 = ((1,1),(1,1))
        point2 = ((2,2),(2,2))
        comparaison = compare_line_verticaly(point1, point2)
        self.assertEqual(comparaison, -1)

    def test_compare_line_verticaly_fist_after_second(self):
        """
        if the first point is after the second one the function return 1
        """
        point1 = ((2,2),(2,2))
        point2 = ((1,1),(1,1))
        comparaison = compare_line_verticaly(point1, point2)
        self.assertEqual(comparaison, 1)

    def test_compare_line_verticaly_fist_equal_second(self):
        """
        if the first point is on the same vertical as the second one the function return 0
        """
        point1 = ((1,1),(1,1))
        point2 = ((1,1),(1,1))
        comparaison = compare_line_verticaly(point1, point2)
        self.assertEqual(comparaison, 0)

class TestMeasureDistance(unittest.TestCase):
    """
    class to test function measure_distance
    """

    def test_measure_distance(self):
        """
        check if the measure is correct
        """
        point1 = (0, 4)
        point2 = (3, 0)
        distance = measure_distance(point1, point2)
        self.assertEqual(distance, 5)

if __name__ == '__main__':
    unittest.main()

import unittest

def my_function(var):
    return var

class Test(unittest.TestCase):
    def test_prop_one(self):
        self.assertEqual(my_function(1),1)
        self.assertEqual(my_function(5),5)

    def test_prop_two(self):
        self.assertEqual(type(my_function(2)),int)

if __name__ == '__main__':
    unittest.main()
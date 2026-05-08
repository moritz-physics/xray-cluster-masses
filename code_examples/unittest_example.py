import unittest

def adding(number1,number2):
  """
  Just a function adding the first and second argument.
  """
  return number1+number2

class AddingTest(unittest.TestCase):
  def test_adding(self):
    """
    We are testing the function adding.
    """
    self.assertEqual(5,adding(2,3))
    self.assertEqual(3.,adding(-1.,4))

  def test_adding_fail(self):
    """
    We are testing the function adding, but this time with a failing test.
    """
    self.assertEqual(6,adding(2,3))
    self.assertEqual(4.,adding(-1.,4))

if __name__ == "__main__":
  unittest.main()

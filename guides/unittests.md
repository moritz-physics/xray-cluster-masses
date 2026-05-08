## Unittests guideline

This is to provide a short guideline on how you can implement unittests on your code.

In this example we test our code for extracting photons. In particular we check whether all photons are included for a particular cluster. This checks whether the extraction size was set correctly.

unittests are built into python's standard library. To use them we can proceed as follows:

```python
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

if __name__ == "__main__":
  unittest.main()

```

This code can also be found in the unittest_example.py file where an additional test has been added which fails when running the test via

```python
python unittest_example.py
```

The next step is how to set up this test such that it runs automatically when pushing your code changes to gitlab.

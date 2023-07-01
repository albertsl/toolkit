First install pytest with:

`pip install pytest`

Then create test files for each file in our project. The file name must start with `test_`

Inside each file we will write tests using this structure:

```
from main import function1, function2, function3, etc

def test_function1:
    assert function1(x) == 3
```

Finally, we run the `pytest` command in the root folder of our project, and it automatically will find all test files and run all tests and show the result.
import pytest
from main import suma, filewrite

def test_suma():
    assert suma(2, 2) == 4

@pytest.mark.parametrize(
    "input_a, input_b, expected",
    [
        (3, 2, 5),
        (2, 3, 5),
        (suma(2, 3), 5, 10),
        (2, suma(3, 5), 10)
    ]
)
def test_suma_multi(input_a, input_b, expected):
    assert suma(input_a, input_b) == expected

def test_filewrite(tmpdir):
    text = "python testing"
    file = f"{tmpdir}/test.txt"
    filewrite(file, text)

    with open(file) as f:
        text_out = f.read()
    
    assert text_out == text
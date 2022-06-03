import numpy as np


def uniquify(data):
    """
    Return list of each string appearing at least once in strs, in order.
    (probably faster way to do this)
    """
    seen = {}
    r = []
    for d in data:
        if d not in seen:
            seen[d] = True
            r.append(d)
    return r


def test_unquify():
    test_input = [1, 2, 2, 2, 2, 3, 3, 1, 1, 2, 3, 1, 2, 2, 4, 2, 3, 2, 4, 2, 2, 1]
    correct_output = [1, 2, 3, 4]
    output = uniquify(test_input)
    assert np.array_equal(correct_output, output), "Mismatch:  %s != %s " % (test_input, output)


if __name__ == "__main__":
    test_unquify()
    print("All tests pass.")

"""Constants and other basic utility methods."""
import numpy as np

EPSILON4 = 1e-4


def sample_from_dict(dictionary, np_random=None):
    if not np_random:
        np_random = np.random
    rn = np_random.random()
    acu = 0.0
    for key in dictionary:
        acu += dictionary[key]
        if rn <= acu + EPSILON4:
            return key
    raise Exception("Total probability is less than one (t<1).")


def sample_uniform_from_list(li, np_random=None):
    if not np_random:
        np_random = np.random
    rn = np_random.randint(0, len(li))
    return li[rn]


def sample_location(width, height, np_random=None):
    if not np_random:
        np_random = np.random
    x = np_random.randint(0, width)
    y = np_random.randint(0, height)
    return x, y

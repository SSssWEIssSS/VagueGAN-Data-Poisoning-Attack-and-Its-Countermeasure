from matplotlib import pyplot as plt
from skimage.util import random_noise
from PIL import Image
import numpy as np
def apply_class_label_replacement(X, Y, replacement_method):

    return (X, replacement_method(Y, set(Y)))

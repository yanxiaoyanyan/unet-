import os
import cv2
import numpy as np


def image_process(lib, height, width):
    image = os.listdir(lib)
    data = np.empty((len(image), height, width))
    for i, name in enumerate(image):
        im = cv2.imread(lib + name, cv2.IMREAD_UNCHANGED)
        im = cv2.resize(im, dsize=(width, height), interpolation=cv2.INTER_LANCZOS4)
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        data[i] = im
    return data





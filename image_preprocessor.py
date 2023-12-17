# Made with the use of Github Co-Pilot
# https://github.com/johnloane/st_23_sd4a/blob/dd3c950d6390db3e9b29fe5f05ecd0a65776697b/mnist_dnn.py#L87

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10, cifar100


def preprocess_images(images):
    # Preprocessing steps like grayscale conversion, normalization, reshaping, Gaussian blur, equalizing histogram etc.
    gray_images  = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    equalized_images = [cv2.equalizeHist(img) for img in gray_images]
    images_array = np.array(equalized_images).reshape((-1, 32, 32, 1))
    images_normalized = images_array.astype('float32') / 255.0
    images = np.expand_dims(images, axis=-1)
    blurred_images = np.array([cv2.GaussianBlur(img, (5, 5), 0) for img in images_normalized])
    return blurred_images
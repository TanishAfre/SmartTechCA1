# Made with the use of Github Co-Pilot

import numpy as np
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2

def load_cifar_datasets():
    # Load CIFAR-10 and CIFAR-100 datasets
    (cifar10_train_images, cifar10_train_labels), _ = cifar10.load_data()
    (cifar100_train_images, cifar100_train_labels), _ = cifar100.load_data()
    return cifar10_train_images, cifar10_train_labels, cifar100_train_images, cifar100_train_labels



if __name__ == "__main__":
    cifar10_train_images, cifar10_train_labels, cifar100_train_images, cifar100_train_labels = load_cifar_datasets()
    

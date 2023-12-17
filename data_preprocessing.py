# Made with the use of Github Co-Pilot

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10, cifar100


def filter_and_combine_datasets(cifar10_train_images, cifar10_train_labels, cifar100_train_images, cifar100_train_labels):
    # Define the class indices for CIFAR-10 and CIFAR-100
    cifar10_classes = {'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'horse': 7, 'truck': 9}
    cifar100_classes = {'cattle': 11, 'fox': 34, 'baby': 2, 'boy': 11, 'girl': 35, 'man': 44, 'woman': 98,
                        'rabbit': 65, 'squirrel': 78, 'trees': 84, 'bicycle': 8, 'bus': 13,
                        'motorcycle': 48, 'pickup truck': 58, 'train': 95, 'lawn-mower': 48, 'tractor': 86}

    # Filter the datasets for the relevant classes and combine them with class names as labels
    cifar10_filtered_images = []
    cifar10_filtered_labels = []
    for img, lbl in zip(cifar10_train_images, cifar10_train_labels):
        if lbl[0] in cifar10_classes.values():
            cifar10_filtered_images.append(img)
            cifar10_filtered_labels.append(list(cifar10_classes.keys())[list(cifar10_classes.values()).index(lbl[0])])

    cifar100_filtered_images = []
    cifar100_filtered_labels = []
    for img, lbl in zip(cifar100_train_images, cifar100_train_labels):
        if lbl[0] in cifar100_classes.values():
            cifar100_filtered_images.append(img)
            cifar100_filtered_labels.append(list(cifar100_classes.keys())[list(cifar100_classes.values()).index(lbl[0])])

    combined_images = cifar10_filtered_images + cifar100_filtered_images
    combined_labels = cifar10_filtered_labels + cifar100_filtered_labels

    return combined_images, combined_labels
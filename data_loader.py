import numpy as np
from tensorflow.keras.datasets import cifar10, cifar100

def load_cifar_datasets():
    # Load CIFAR-10 and CIFAR-100 datasets
    (cifar10_train_images, cifar10_train_labels), _ = cifar10.load_data()
    (cifar100_train_images, cifar100_train_labels), _ = cifar100.load_data()
    return cifar10_train_images, cifar10_train_labels, cifar100_train_images, cifar100_train_labels
# Made with the use of Github Co-Pilot
 
import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_sample_images(images):
    plt.figure(figsize=(10, 10))
    for i in range(min(len(images), 24)):
        plt.subplot(5, 5, i + 1)
        # Check if the image is grayscale or color
        if len(images[i].shape) == 3 and images[i].shape[2] == 3:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i].reshape((32, 32)), cmap='gray')
        plt.axis('off')
    plt.show()

def display_training_images(images):
    plt.figure(figsize=(10, 10))
    for i in range(min(len(images), 24)):
        plt.subplot(5, 5, i + 1)
        # Check if the image is grayscale or color
        if len(images[i].shape) == 3 and images[i].shape[2] == 3:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i].reshape((32, 32)), cmap='gray')
        plt.axis('off')
    plt.show()

    # display 10 random augmented images
def plot_sample_augmented_images(images):
    plt.figure(figsize=(10, 10))
    for i in range(min(len(images), 10)):
        plt.subplot(5, 5, i + 1)
        # Check if the image is grayscale or color
        if len(images[i].shape) == 3 and images[i].shape[2] == 3:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i].reshape((32, 32)), cmap='gray')
        plt.axis('off')
    plt.show()
    

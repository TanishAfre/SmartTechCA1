# Made with the use of Github Co-Pilot
# Followed https://github.com/johnloane/SD4PythonRefresh23/blob/c2840d968f0673417b3f2dffccf2a48cb187b32c/gts_cnn.py#L140

import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_images(images):
    # Create an instance of ImageDataGenerator with specified augmentation parameters.
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        rotation_range=10
    )


    augmented_images = []
    for img in images:
        # Ensure the image is in the correct format, e.g., convert to RGB if necessary
        if len(img.shape) == 2 or img.shape[2] == 1:  # Grayscale or single channel
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Apply data augmentation
        img = datagen.random_transform(img)
        augmented_images.append(img)

    augmented_images = np.array(augmented_images)

    return augmented_images

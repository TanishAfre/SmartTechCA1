import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image

def zoomed_image(images):
    zoomed_images = []
    for img in images:
        zoomed_images.append(cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC))
    return np.array(zoomed_images)

def pan_image(images):
    panned_images = []
    for img in images:
        # The image is first padded with 8 pixels on each side
        padded_img = cv2.copyMakeBorder(img, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        # Then, a random 64x64 crop is taken from the padded image
        random_crop = np.random.randint(0, 16)
        cropped_img = padded_img[random_crop:random_crop + 64, random_crop:random_crop + 64]
        panned_images.append(cropped_img)
    return np.array(panned_images)

def darken_image(images):
    darkened_images = []
    for img in images:
        # The image is first converted to HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # Then, the V channel is multiplied by a random number between 0.5 and 1.0
        random_brightness = np.random.uniform(0.5, 1.0)
        hsv_img[:, :, 2] = hsv_img[:, :, 2] * random_brightness
        # Finally, the image is converted back to RGB
        rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        darkened_images.append(rgb_img)
    return np.array(darkened_images)

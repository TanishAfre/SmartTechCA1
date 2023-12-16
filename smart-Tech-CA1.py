# Made with the use of Github Co-Pilot

# main.py
from data_loader import load_cifar_datasets
from data_preprocessing import filter_and_combine_datasets
from image_preprocessor import preprocess_images
from data_visualization import plot_sample_images, display_training_images
from data_models import create_and_train_model, underfitting_model, overfitting_model
from data_augmentaion import zoomed_image, pan_image, darken_image

import numpy as np

if __name__ == "__main__":
    cifar10_train_images, cifar10_train_labels, cifar100_train_images, cifar100_train_labels = load_cifar_datasets()

    combined_images, combined_labels = filter_and_combine_datasets(cifar10_train_images, cifar10_train_labels,
                                                                     cifar100_train_images, cifar100_train_labels)
    
    plot_sample_images(combined_images)

    preprocessed_images = preprocess_images(combined_images) 

    display_training_images(preprocessed_images)

    #underfitting_model(preprocessed_images, combined_labels)

    #overfitting_model(preprocessed_images, combined_labels)

    # call the data augmentation functions
    zoomed_images = zoomed_image(preprocessed_images)
    panned_images = pan_image(preprocessed_images)
    darkened_images = darken_image(preprocessed_images)

    # display the augmented images
    #display_zoomed_images(zoomed_images)
    #display_panned_images(panned_images)
    #display_darkened_images(darkened_images)


    # combine the augmented images with the original images
    combined_images = np.concatenate((preprocessed_images, zoomed_images, panned_images, darkened_images))
    combined_labels = np.concatenate((combined_labels, combined_labels, combined_labels, combined_labels))


    model = create_and_train_model(preprocessed_images, combined_labels)

    model.save('model.h5')

# Made with the use of Github Co-Pilot

# main.py
from data_loader import load_cifar_datasets
from data_preprocessing import filter_and_combine_datasets
from image_preprocessor import preprocess_images
from data_visualization import plot_sample_images, display_training_images, plot_sample_augmented_images
from data_models import create_and_train_model, underfitting_model, overfitting_model
from data_augmentaion import augment_images

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

    # test to see dimensions of each array
    # print("Preprocessed images shape:", preprocessed_images.shape) - (42500, 32, 32)
    # print("Augmented images shape:", augmented_images.shape) - (42500, 32, 32, 3)
    #hence changing the input shape of the model to (32, 32, 0) instead of (32, 32, 1)
    preprocessed_images = preprocessed_images.reshape((42500, 32, 32, 1))
    # call the data augmentation functions
    augmented_images = augment_images(preprocessed_images)

    plot_sample_augmented_images(augmented_images)

    model = create_and_train_model(preprocessed_images, combined_labels)

    model.save('model.h5')

    # printing model summary 
    model = leNet_model()
    print(model.summary())

    





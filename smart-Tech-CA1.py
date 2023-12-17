# Made with the use of Github Co-Pilot

# main.py
from data_loader import load_cifar_datasets
from data_preprocessing import filter_and_combine_datasets
from image_preprocessor import preprocess_images
from data_visualization import plot_sample_images, display_training_images, plot_sample_augmented_images
from data_models import create_and_train_model, underfitting_model, overfitting_model
from data_augmentaion import augment_images

import numpy as np
import matplotlib.pyplot as plt

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

    # Split data into training and validation sets
    # For example, using a 80-20 split:
    combined_labels = np.array(combined_labels)

    split_index = int(0.8 * len(preprocessed_images))
    train_images = preprocessed_images[:split_index]
    train_labels = combined_labels[:split_index]
    val_images = preprocessed_images[split_index:]
    val_labels = combined_labels[split_index:]


    # Trying to get test to work
    # val_labels = np.array(val_labels)
    # assert isinstance(val_labels, np.ndarray), "Validation labels must be a NumPy array"

    #validation_data = (val_images, val_labels)

    #assert isinstance(validation_data[0], np.ndarray), "Validation images must be a NumPy array"
    #assert isinstance(validation_data[1], np.ndarray), "Validation labels must be a NumPy array"

    #model, history = create_and_train_model(train_images, train_labels, validation_data=(val_images, val_labels))
    model = create_and_train_model(train_images, train_labels)

    model.save('model.h5')

    # printing model summary 
    print(model.summary())
    
    # plotting the accuracy and loss of the model
    # plt.figure(figsize=(10, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper left')

    # Testing model
    url = "https://raw.githubusercontent.com/YoongiKim/CIFAR-10-images/master/test/truck/0054.jpg" # Is a truck
    response = requests.get(url, stream=True)
    img = Image.open(response.raw)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocess_images(img)
    img = img.reshape(1, 32, 32, 1)
    print("predicted sign: "+ str(model.predict_classes(img), axis=-1))





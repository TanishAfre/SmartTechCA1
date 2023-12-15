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

def preprocess_images(images):
    # Preprocessing steps like grayscale conversion, normalization, reshaping, Gaussian blur, equalizing histogram etc.
    gray_images  = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    equalized_images = [cv2.equalizeHist(img) for img in gray_images]
    images_array = np.array(equalized_images).reshape((-1, 32, 32, 1))
    images_normalized = images_array.astype('float32') / 255.0
    images = np.expand_dims(images, axis=-1)
    blurred_images = np.array([cv2.GaussianBlur(img, (5, 5), 0) for img in images_normalized])
    return blurred_images

def plot_sample_images(images):
    # Plot sample images
    plt.figure(figsize=(10, 10))
    for i in range(24):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].reshape((32, 32)), cmap='gray')
        plt.axis('off')
    plt.show()

def plot_label_histograms(labels):
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(15, 5))

    plt.bar(unique_labels, label_counts)
    plt.xlabel('Class Name')
    plt.ylabel('Count')
    plt.title('Label Distribution (Combined CIFAR-10 and CIFAR-100)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # equalize the histogram
    plt.figure(figsize=(15, 5))
    plt.bar(unique_labels, label_counts / sum(label_counts))
    plt.xlabel('Class Name')
    plt.ylabel('Count')
    plt.title('Label Distribution (Combined CIFAR-10 and CIFAR-100)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def display_training_images(images, labels):
    # Display the first 100 images with labels
    plt.figure(figsize=(15, 15))
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.imshow(images[i].reshape((32, 32)), cmap='gray')
        plt.axis('off')
        plt.title(labels[i])
    plt.show()


if __name__ == "__main__":
    cifar10_train_images, cifar10_train_labels, cifar100_train_images, cifar100_train_labels = load_cifar_datasets()
    
    combined_images, combined_labels = filter_and_combine_datasets(cifar10_train_images, cifar10_train_labels, cifar100_train_images, cifar100_train_labels)
    combined_images = preprocess_images(combined_images)
    
    # Plot sample images
    plot_sample_images(combined_images)
    
    # Plot combined label histogram
    plot_label_histograms(combined_labels)

    display_training_images(combined_images, combined_labels) 

# Plot the first image in the combined dataset
label_mapping = {label: idx for idx, label in enumerate(set(combined_train_labels))}
combined_train_labels = np.array([label_mapping[label] for label in combined_train_labels])

print(combined_train_images[0])

ann = tf.keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(22, activation='softmax')  # Changed to softmax for multi-class classification
])
# Compile the model
ann.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Fit the model
ann.fit(combined_train_images, combined_train_labels, epochs=15, batch_size=32)  # Ensure batch_size is appropriate

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np


def underfitting_model(combined_train_images, combined_train_labels):

    label_mapping = {label: idx for idx, label in enumerate(set(combined_train_labels))}
    combined_train_labels = np.array([label_mapping[label] for label in combined_train_labels])
    
    ann = tf.keras.Sequential([
    layers.Flatten(input_shape=(32, 32, 1)),
    layers.Dense(30, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(22, activation='softmax')  # Changed to softmax for multi-class classification
    ])
# Compile the model
    ann.compile(optimizer='SGD',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Fit the model
    ann.fit(combined_train_images, combined_train_labels, epochs=5, batch_size=32)  # Ensure batch_size is appropriate

    return ann

def overfitting_model(combined_train_images, combined_train_labels, epochs=15, batch_size=32):

    label_mapping = {label: idx for idx, label in enumerate(set(combined_train_labels))}
    combined_train_labels = np.array([label_mapping[label] for label in combined_train_labels])

    # Create a more complex model
    overfitting_model = tf.keras.Sequential([
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),  # Add dropout for regularization
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(22, activation='softmax')
    ])

    # Compile the model
    overfitting_model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

    # Fit the model with more epochs
    overfitting_model.fit(combined_train_images, combined_train_labels, epochs=epochs, batch_size=batch_size)

    return overfitting_model


def create_and_train_model(combined_train_images, combined_train_labels, epochs=1, batch_size=32):
    
    # Trying to get test to work
    #def create_and_train_model(combined_train_images, combined_train_labels, validation_data=None, epochs=1, batch_size=32):

    label_mapping = {label: idx for idx, label in enumerate(set(combined_train_labels))}
    combined_train_labels = np.array([label_mapping[label] for label in combined_train_labels])
    # Create the model
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
        layers.Dense(22, activation='softmax')
    ])

    # Compile the model
    ann.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Trying to get test to work
    #history = ann.fit(combined_train_images, combined_train_labels, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    # Fit the model
    ann.fit(combined_train_images, combined_train_labels, epochs=epochs, batch_size=batch_size)


    return ann
    # Trying to get test to work
    #return ann, history

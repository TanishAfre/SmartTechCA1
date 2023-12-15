# Made with the use of Github Co-Pilot

# main.py
from data_loader import load_cifar_datasets
from data_preprocessing import filter_and_combine_datasets
from image_preprocessor import preprocess_images
from data_visualization import plot_sample_images, display_training_images
from data_models import create_and_train_model, underfitting_model

if __name__ == "__main__":
    cifar10_train_images, cifar10_train_labels, cifar100_train_images, cifar100_train_labels = load_cifar_datasets()

    combined_images, combined_labels = filter_and_combine_datasets(cifar10_train_images, cifar10_train_labels,
                                                                     cifar100_train_images, cifar100_train_labels)
    
    plot_sample_images(combined_images)

    #print_sample_trees(combined_images, combined_labels)

    preprocessed_images = preprocess_images(combined_images)

    display_training_images(preprocessed_images)

    model = create_and_train_model(preprocessed_images, combined_labels)

    model.save('model.h5')

import tensorflow as tf
from tensorflow.karas import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()


#Test Tensorflow in CPU mode (e.g., HPC headnode)
import tensorflow as tf
import pathlib
print("TensorFlow version:", tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
num_gpu = len(physical_devices)
print("Num GPUs:", num_gpu)

import tensorflow_datasets as tfds #pip install tensorflow_datasets
datasets, info = tfds.load('mnist', with_info=True, as_supervised=True)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#Dataset mnist downloaded and prepared to ~/tensorflow_datasets/mnist/3.0.1.

kerasdata = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = kerasdata.load_data()

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
imagefolderpath = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
print('Flower dataset downloaded to:', imagefolderpath)#Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
data_dir = pathlib.Path(imagefolderpath)
imagepattern='*/*.'+'jpg'
image_count = len(list(data_dir.glob(imagepattern)))
print("Flower dataset imagecount:",image_count)#3670
#Flower dataset downloaded to: /home/010796032/.keras/datasets/flower_photos
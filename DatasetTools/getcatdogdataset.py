import tensorflow as tf
import pathlib
import PIL
import PIL.Image
import os

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
dirpath=os.path.dirname(path_to_zip)
print(dirpath)#/home/lkk/.keras/datasets

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')#'/home/lkk/.keras/datasets/cats_and_dogs_filtered
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_dir = pathlib.Path(train_dir)
print(train_dir) #/home/lkk/.keras/datasets/cats_and_dogs_filtered/train
image_count = len(list(train_dir.glob('*/*.jpg')))
print(image_count) #2000
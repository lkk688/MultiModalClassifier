import tensorflow as tf
import pathlib
import PIL
import PIL.Image
import numpy as np

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname='flower_photos', 
                                   untar=True)
data_dir = pathlib.Path(data_dir)
print(data_dir) #/home/kaikai/.keras/datasets/flower_photos
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count) #3670

#Display one image:
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
#['dandelion', 'roses', 'tulips', 'sunflowers', 'daisy']
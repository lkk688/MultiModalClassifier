import tensorflow as tf
import pathlib
import PIL
import PIL.Image
import numpy as np
import math

class_names = None
IMG_height = 180
IMG_width = 180

def decode_jpeg_and_label(filename):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize the image to the desired size.
    image = tf.image.resize(image, [IMG_height, IMG_width])
    # parse flower name from containing directory
    label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
    label = label.values[-2]
    return image, label

def show_oneimage(image_batch, label_batch):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image_batch)
    plt.title(label_batch.numpy().title())
    fig.savefig('./outputs/plotoneimage.png')#


def main():
    GCS_PATTERN = 'gs://flowers-public/*/*.jpg'
    AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API
    #GCS_OUTPUT = 'gs://flowers-public/tfrecords-jpeg-192x192-2/flowers'  # prefix for output file names
    SHARDS = 16
    TARGET_SIZE = [IMG_height, IMG_width]
    global class_names
    class_names = [b'daisy', b'dandelion', b'roses', b'sunflowers', b'tulips'] # do not change, maps to the labels in the data (folder names)

    nb_images = len(tf.io.gfile.glob(GCS_PATTERN))
    shard_size = math.ceil(1.0 * nb_images / SHARDS)
    print("Pattern matches {} images which will be rewritten as {} .tfrec files containing {} images each.".format(nb_images, SHARDS, shard_size))

    filenames = tf.data.Dataset.list_files(GCS_PATTERN, seed=35155) # This also shuffles the images
    dataset1 = filenames.map(decode_jpeg_and_label, num_parallel_calls=AUTO)

    image_batch, label_batch = next(iter(dataset1))
    print(image_batch.shape)
    print(label_batch)

if __name__ == '__main__':
    main()
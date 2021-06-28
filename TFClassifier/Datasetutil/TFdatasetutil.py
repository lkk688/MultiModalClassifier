import tensorflow as tf
import numpy as np

def loadtfds(name='mnist'):
    import tensorflow_datasets as tfds
    datasets, info = tfds.load(name, with_info=True, as_supervised=True) #downloaded and prepared to /home/lkk/tensorflow_datasets/mnist/3.0.1.
    train, test = datasets['train'], datasets['test'] 

    # You can also do info.splits.total_num_examples to get the total
    # number of examples in the dataset.

    num_train_examples = info.splits['train'].num_examples
    num_test_examples = info.splits['test'].num_examples

    train_data=train.map(scale)
    test_data=test.map(scale)
    return train_data, test_data, num_train_examples, num_test_examples

def loadkerasdataset(name='fashionMNIST'):
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    #60000, 28, 28, 
    #(60000,)
    # Adding a dimension to the array -> new shape == (28, 28, 1)
    # We are doing this because the first layer in our model is a convolutional
    # layer and it requires a 4D input (batch_size, height, width, channels).
    # batch_size dimension will be added later on.
    train_images = train_images[..., None]#(60000, 28, 28, 1)
    test_images = test_images[..., None]#(10000, 28, 28, 1)

    # Getting the images in [0, 1] range.
    train_images = train_images / np.float32(255)
    test_images = test_images / np.float32(255)

    #The given tensors are sliced along their first dimension.
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    num_train=len(train_dataset)#60000
    num_test=len(test_dataset)
    return train_dataset, test_dataset, num_train, num_test


# Pixel values, which are 0-255, have to be normalized to the 0-1 range. Define this scale in a function.
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label


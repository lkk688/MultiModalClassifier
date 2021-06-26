import tensorflow as tf

def loadtfds(name='mnist'):
    import tensorflow_datasets as tfds
    datasets, info = tfds.load(name, with_info=True, as_supervised=True) #downloaded and prepared to /home/lkk/tensorflow_datasets/mnist/3.0.1.
    train, test = datasets['train'], datasets['test'] 

    # You can also do info.splits.total_num_examples to get the total
    # number of examples in the dataset.

    num_train_examples = info.splits['train'].num_examples
    num_test_examples = info.splits['test'].num_examples
    return train, test, num_train_examples, num_test_examples

# Pixel values, which are 0-255, have to be normalized to the 0-1 range. Define this scale in a function.


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label


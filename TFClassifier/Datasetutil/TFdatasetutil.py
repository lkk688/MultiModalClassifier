import tensorflow as tf
import numpy as np

#from .Visutil import plot25images, plot9imagesfromtfdataset
#import Visutil
#import Datasetutil.Visutil as Visutil
#from Datasetutil.Visutil import Visutil
from TFClassifier.Datasetutil.Visutil import plot25images, plot9imagesfromtfdataset

AUTO = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 32
IMG_height = 180
IMG_width = 180

def loadTFdataset(name, type, path='/home/lkk/.keras/datasets/flower_photos', img_height=180, img_width=180, batch_size=32):
    global BATCH_SIZE
    BATCH_SIZE=batch_size
    global IMG_height, IMG_width
    IMG_height = img_height
    IMG_width = img_width

    if type=='tfds':
        train_data, test_data, num_train_examples, num_test_examples, class_names, imageshape = loadtfds(name)
        train_ds, val_ds = setBatchtoTFdataset(train_data, test_data, batch_size)
        plot9imagesfromtfdataset(train_ds, class_names)
    elif type=='kerasdataset':
        train_data, test_data, num_train_examples, num_test_examples, class_names, imageshape = loadkerasdataset(name)
        train_ds, val_ds = setBatchtoTFdataset(train_data, test_data, batch_size)
    elif type=='imagefolder':
        train_ds, val_ds, class_names=loadimagefolderdataset(name, path, img_height, img_width, batch_size)
    else:
        print('Data tpye not supported')
        exit()
    return train_ds, val_ds, class_names, imageshape

def loadtfds(name='mnist'):
    import tensorflow_datasets as tfds
    datasets, info = tfds.load(name, with_info=True, as_supervised=True) #downloaded and prepared to /home/lkk/tensorflow_datasets/mnist/3.0.1.    
    train, test = datasets['train'], datasets['test'] 

    #use the TFDS API to visualize how our images look like
    fig = tfds.show_examples(train, info)
    fig.savefig('tfdsvis.png')

    ds = train.take(1)#https://www.tensorflow.org/datasets/overview#installation
    for image, label in tfds.as_numpy(ds):
        print(np.min(image), np.max(image)) #0, 255
        print(type(image), type(label), label) #<class 'numpy.ndarray'> <class 'numpy.int64'> 4


    # You can also do info.splits.total_num_examples to get the total
    # number of examples in the dataset.

    num_train_examples = info.splits['train'].num_examples
    num_test_examples = info.splits['test'].num_examples

    class_names = info.features['label'].names
    print("Num classes: " + str(info.features['label'].num_classes))
    imageshape =info.features['image'].shape
    global IMG_height, IMG_width
    IMG_height = imageshape[0]
    IMG_width = imageshape[1]
    print(imageshape)#(28, 28, 1)

    train_data=train.map(scale, num_parallel_calls=AUTO)#scale to 0-1
    test_data=test.map(scale, num_parallel_calls=AUTO)

    return train_data, test_data, num_train_examples, num_test_examples, class_names, imageshape

def loadkerasdataset(name='fashionMNIST'):
    if name=='fashionMNIST':
        kerasdata = tf.keras.datasets.fashion_mnist
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif name=='cifar10':#(50000, 32, 32, 3)
        kerasdata = tf.keras.datasets.cifar10 #ownloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        print("This dataset is not supported:", name)
        exit()

    (train_images, train_labels), (test_images, test_labels) = kerasdata.load_data()
    #60000, 28, 28,  The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255
    #(60000,) The labels are an array of integers, ranging from 0 to 9

    # Getting the images in [0, 1] range.
    train_images = train_images / np.float32(255)
    test_images = test_images / np.float32(255)

    plot25images(train_images, train_labels, class_names)

    # Adding a dimension to the array -> new shape == (28, 28, 1)
    # We are doing this because the first layer in our model is a convolutional
    # layer and it requires a 4D input (batch_size, height, width, channels).
    # batch_size dimension will be added later on.
    train_images = train_images[..., None]#(60000, 28, 28, 1)
    test_images = test_images[..., None]#(10000, 28, 28, 1)

    imageshape=train_images.shape[1:]

    #The given tensors are sliced along their first dimension.
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    num_train=len(train_dataset)#60000
    num_test=len(test_dataset)

    return train_dataset, test_dataset, num_train, num_test, class_names, imageshape

def setBatchtoTFdataset(train_data, test_data, BATCH_SIZE=32, BUFFER_SIZE=10000):
    # Apply this function to the training and test data, shuffle the training data, and batch it for training.
    # This dataset fills a buffer with buffer_size elements, then randomly samples elements from this buffer, replacing the selected elements with new elements. For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
    # train_ds = train_data.map(scale).cache().shuffle(
    #     BUFFER_SIZE).batch(BATCH_SIZE)
    # val_ds = test_data.map(scale).batch(BATCH_SIZE)
    train_ds = train_data.cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)
    val_ds = test_data.batch(BATCH_SIZE)
    return train_ds, val_ds

# Pixel values, which are 0-255, have to be normalized to the 0-1 range. Define this scale in a function.
# def scale(image, label):
#     image = tf.cast(image, tf.float32)
#     image /= 255
#     return image, label

@tf.function
def scale_resize_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # equivalent to dividing image pixels by 255
    image = tf.image.resize(image, (IMG_height, IMG_width)) # Resizing the image to  dimention
    return (image, label)

@tf.function
def scale(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return (image, label)

@tf.function
def random_crop(images, labels):
    boxes = tf.random.uniform(shape=(len(images), 4))
    box_indices = tf.random.uniform(shape=(len(images),), minval=0, maxval=BATCH_SIZE, dtype=tf.int32)
    images = tf.image.crop_and_resize(images, boxes, box_indices, (IMG_height,IMG_width))
    return images, labels

def loadimagefolderdataset(name, imagefolderpath='~/.keras/datasets/flower_photos', imageformat='jpg', img_height=180, img_width=180, batch_size=32):
    import pathlib
    data_dir = pathlib.Path(imagefolderpath)
    if imageformat=='jpg' or imageformat=='png':
        imagepattern='*/*.'+imageformat
        image_count = len(list(data_dir.glob(imagepattern)))
        if image_count<=0 and name=='flower':
            dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
            imagefolderpath = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)##/home/lkk/.keras/datasets/flower_photos
            print('Flower dataset downloaded to:', imagefolderpath)#Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
            data_dir = pathlib.Path(imagefolderpath)
            image_count = len(list(data_dir.glob(imagepattern)))
            print(image_count)#3670
        elif image_count>0:
            print("image_count: ", image_count)
        else:
            print('Image folder does not have images')
            exit()
    else:
        print("file format not supported")
        exit()
    
    #create dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    class_names = train_ds.class_names
    print("class_names:",class_names)
    #see the length of each dataset as follows:
    num_train_batch=tf.data.experimental.cardinality(train_ds).numpy()
    num_test_batch=tf.data.experimental.cardinality(val_ds).numpy()

    #manually iterate over the dataset and retrieve batches of images:
    #This is a batch of 32 images of shape 180x180x3 (the last dimension referes to color channels RGB). The label_batch is a tensor of the shape (32,), these are corresponding labels to the 32 images.
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    plot9imagesfromtfdataset(train_ds, class_names)
    return train_ds, val_ds, class_names


def test_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"

if __name__ == "__main__":
    test_sum()
    train_ds, val_ds, class_names, img_size = loadTFdataset('mnist', 'tfds')
    print(len(train_ds))
    print("Everything passed")
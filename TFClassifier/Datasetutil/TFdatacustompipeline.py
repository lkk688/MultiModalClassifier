
# For finer grain control, you can write your own input pipeline using tf.data: tf.data.Dataset.list_files
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import glob
import matplotlib.pyplot as plt

class_names = None
IMG_height = 180
IMG_width = 180
BATCH_SIZE = 32
ONE_HOT_encoding = False 
AUTOTUNE = tf.data.AUTOTUNE #tf.data.experimental.AUTOTUNE

def checkdataset(list_ds, itemcount=2):
    for f in list_ds.take(itemcount):
        print(f.numpy())
        print(f.numpy().decode('utf-8'))

def checkimglabeldataset(train_ds, itemcount=2):
    for image, label in train_ds.take(itemcount):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names ##one hot encoding, return array([False, False,  True, False, False])
    if ONE_HOT_encoding==True:
        return one_hot
    else:
        return tf.argmax(one_hot) # Integer encode the label


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size
    return tf.image.resize(img, [IMG_height, IMG_width])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

#Configure dataset for performance
def configure_for_performance(ds, AUTOTUNE):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0, 2)
    image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
    #https://www.tensorflow.org/tutorials/images/data_augmentation
    return image, label  

def plot25imagesfromds(dataset):
    image_batch, label_batch = next(iter(dataset))
    fig = plt.figure(figsize=(10, 10))
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))#pixel range from 0-255
        if ONE_HOT_encoding==True:
            # to return true indices. 
            res = [i for i, val in enumerate(label_batch[i]) if val] 
            label = res[0]
        else:
            label = label_batch[i]
        plt.title(class_names[label])
        plt.axis("off")
        if i>= BATCH_SIZE:
            break
    fig.savefig('./outputs/plot25imagesfromds.png')#

def plotoneimagefromds(dataset):
    image_batch, label_batch =next(iter(dataset))
    fig = plt.figure(figsize=(10,10))
    plt.imshow(image_batch)
    plt.title(class_names[label_batch.numpy()])
    fig.savefig('./outputs/plotoneimagefromds.png')#

# Pixel values, which are 0-255, have to be normalized to the 0-1 range. Define this scale in a function.
@tf.function
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

def processdir(data_dir_str='/home/lkk/.keras/datasets/flower_photos'):
    data_dir = pathlib.Path(data_dir_str)

    filelist = (data_dir_str+'/*/*.jpg') #(data_dir_str+'/*/*')  # create glob pattern
    datapattern = glob.glob(filelist) #another option: tf.io.gfile.glob
    image_count = len(datapattern)

    global class_names
    class_names = np.array(sorted(
        [item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))

    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
    # Option2: list_ds = tf.data.Dataset.list_files(filelist, shuffle=False) # str(data_dir/'*/*'), filelist

    checkdataset(list_ds, 2)  # check data

    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    checkdataset(list_ds, 2)  # check data

    # Split the dataset into train and validation:
    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    # see the length of each dataset as follows:
    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(val_ds).numpy())

    #Use Dataset.map to create a dataset of image, label pairs:
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    checkimglabeldataset(train_ds,2)

    # train_ds = configure_for_performance(train_ds, AUTOTUNE)
    # val_ds = configure_for_performance(val_ds, AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    plot25imagesfromds(train_ds)#plot 0-255 value image

    train_ds=train_ds.map(scale, num_parallel_calls=AUTOTUNE)#scale to 0-1
    val_ds=val_ds.map(scale, num_parallel_calls=AUTOTUNE)

    for image_batch, labels_batch in train_ds:
        imagetensorshape = image_batch.get_shape().as_list()
        imageshape=imagetensorshape[1:]
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    return train_ds, val_ds, class_names, imageshape

def read_tfrecord(example):
    features = {
        # tf.string means bytestring
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
    }
    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    # convert image to floats in [0, 1] range
    image = tf.cast(image, tf.float32) / 255.0
    # explicit size will be needed for TPU
    IMAGE_SIZE=[IMG_height, IMG_width]
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    class_label = example['class']
    return image, class_label


def load_dataset(filenames):
    # read from TFRecords. For optimal performance, read from multiple
    # TFRecord files at once and set the option experimental_deterministic = False
    # to allow order-altering optimizations.

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    return dataset

def get_batched_dataset(filenames, train=False):
    dataset = load_dataset(filenames)
    dataset = dataset.cache() # This dataset fits in RAM
    if train:
        # Best practices for Keras:
        # Training dataset: repeat then batch
        # Evaluation dataset: do not repeat
        dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE) # prefetch next batch while training (autotune prefetch buffer size)
    # should shuffle too but this dataset was well shuffled on disk already
    return dataset
    # source: Dataset performance guide: https://www.tensorflow.org/guide/performance/datasets

def processtfrecorddir(tfrecordpath='./outputs/TFrecord/'):
    File_pattern = tfrecordpath+'/*.tfrec' #'gs://cmpelkk_imagetest/*.tfrec' #'gs://flowers-public/tfrecords-jpeg-192x192-2/*.tfrec'
    
    VALIDATION_SPLIT = 0.19
    global class_names
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'] # do not change, maps to the labels in the data (folder names)

    # splitting data files between training and validation
    filenames = tf.io.gfile.glob(File_pattern)
    print(len(filenames))

    total_images=3670 #flowers folder 
    split = int(len(filenames) * VALIDATION_SPLIT)
    training_filenames = filenames[split:]
    validation_filenames = filenames[:split]
    print("Pattern matches {} data files. Splitting dataset into {} training files and {} validation files".format(len(filenames), len(training_filenames), len(validation_filenames)))
    validation_steps = int(total_images // len(filenames) * len(validation_filenames)) // BATCH_SIZE
    steps_per_epoch = int(total_images // len(filenames) * len(training_filenames)) // BATCH_SIZE
    print("With a batch size of {}, there will be {} batches per training epoch and {} batch(es) per validation run.".format(BATCH_SIZE, steps_per_epoch, validation_steps))

    train_ds=load_dataset(training_filenames)##scale to 0-1
    val_ds=load_dataset(training_filenames)

    #testimage, testlabel =next(iter(train_ds))
    plotoneimagefromds(train_ds)
    
    # instantiate the datasets
    #train_ds = get_batched_dataset(training_filenames, train=True)
    #val_ds = get_batched_dataset(validation_filenames, train=False)

    train_ds = train_ds.batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    plot25imagesfromds(train_ds)

    #train_ds=train_ds.map(scale, num_parallel_calls=AUTOTUNE)#scale to 0-1
    #val_ds=val_ds.map(scale, num_parallel_calls=AUTOTUNE)

    for image_batch, labels_batch in train_ds:
        imagetensorshape = image_batch.get_shape().as_list()
        imageshape=imagetensorshape[1:]
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    return train_ds, val_ds, class_names, imageshape

def loadTFcustomdataset(name, type, path='/home/lkk/.keras/datasets/flower_photos', img_height=180, img_width=180, batch_size=32):
    global BATCH_SIZE
    BATCH_SIZE=batch_size
    global IMG_height, IMG_width
    IMG_height = img_height
    IMG_width = img_width
    if type=='customdatasetfromfolder':
        #train_ds, val_ds, class_names, imageshape = processdir(path)
        return processdir(path)
    elif type=='customtfrecordfile':
        #train_ds, val_ds, class_names, imageshape = processtfrecorddir(path)
        return processtfrecorddir(path)

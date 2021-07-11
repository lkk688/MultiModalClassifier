
# For finer grain control, you can write your own input pipeline using tf.data: tf.data.Dataset.list_files
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import glob

class_names = None
IMG_height = 180
IMG_width = 180
BATCH_SIZE = 32
ONE_HOT_encoding = False 

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
    import matplotlib.pyplot as plt
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
    AUTOTUNE = tf.data.AUTOTUNE #tf.data.experimental.AUTOTUNE
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    checkimglabeldataset(train_ds,2)

    # train_ds = configure_for_performance(train_ds, AUTOTUNE)
    # val_ds = configure_for_performance(val_ds, AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    plot9imagesfromds(train_ds)#plot 0-255 value image

    train_ds=train_ds.map(scale, num_parallel_calls=AUTOTUNE)#scale to 0-1
    val_ds=val_ds.map(scale, num_parallel_calls=AUTOTUNE)

    for image_batch, labels_batch in train_ds:
        imagetensorshape = image_batch.get_shape().as_list()
        imageshape=imagetensorshape[1:]
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    return train_ds, val_ds, class_names, imageshape

def processtfrecorddir(path):
    return 

def loadTFcustomdataset(name, type, path='/home/lkk/.keras/datasets/flower_photos', img_height=180, img_width=180, batch_size=32):
    global BATCH_SIZE
    BATCH_SIZE=batch_size
    global IMG_height, IMG_width
    IMG_height = img_height
    IMG_width = img_width
    if type=='customdatasetfromfolder':
        train_ds, val_ds, class_names, imageshape = processdir(path)
    elif type=='customtfrecordfile':
        train_ds, val_ds, class_names, imageshape = processtfrecorddir(path)

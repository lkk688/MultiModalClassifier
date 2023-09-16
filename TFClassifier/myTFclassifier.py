from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import pandas as pd
import os
import sys
import math
#import numpy as np
#from matplotlib import pyplot as plt

import numpy as np
import pathlib
import glob
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import time

import configargparse
#import logger

parser = configargparse.ArgParser(description='myTFClassify')
parser.add_argument('--data_path', type=str, default='/home/kaikai/.keras/datasets/flower_photos',
                    help='path to get data')
parser.add_argument('--save_path', type=str, default='./logs',
                    help='path to save the model and logs')
parser.add_argument('--data_type', default='folder', choices=['folder', 'TFrecord'],
                    help='the type of data') #gs://cmpelkk_imagetest/*.tfrec
# network
parser.add_argument('--model_name', default='depth', choices=['disparity', 'depth'],
                    help='the network')
parser.add_argument('--arch', default='Tensorflow', choices=['Tensorflow', 'Pytorch'],
                    help='Model Name, default: Tensorflow.')
parser.add_argument('--GPU', type=bool, default=True,
                    help='use GPU')
parser.add_argument('--TPU', type=bool, default=False,
                    help='use GPU')
# parser.add_argument('--GPU', action='store_true',
#                     help='If true, use  , default: False')
# parser.add_argument('--TPU', action='store_true',
#                     help='If true, use  , default: False')

args = parser.parse_args()

IMAGE_SIZE = [192, 192]
EPOCHS = 30
VALIDATION_SPLIT = 0.19
CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
numclasses = 5


def lrfn(epoch):
    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        if epoch < rampup_epochs:
            lr = (max_lr - start_lr)/rampup_epochs * epoch + start_lr
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        else:
            lr = (max_lr - min_lr) * exp_decay**(epoch -
                                                 rampup_epochs-sustain_epochs) + min_lr
        return lr
    return lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)


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
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    class_label = example['class']
    return image, class_label


def load_dataset(filenames):
    # read from TFRecords. For optimal performance, read from multiple
    # TFRecord files at once and set the option experimental_deterministic = False
    # to allow order-altering optimizations.

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    return dataset


def get_batched_dataset(filenames, train=False):
    dataset = load_dataset(filenames)
    dataset = dataset.cache()  # This dataset fits in RAM
    if train:
        # Best practices for Keras:
        # Training dataset: repeat then batch
        # Evaluation dataset: do not repeat
        dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    # prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.prefetch(AUTO)
    # should shuffle too but this dataset was well shuffled on disk already
    return dataset
    # source: Dataset performance guide: https://www.tensorflow.org/guide/performance/datasets


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == CLASSES
    # Integer encode the label
    return tf.argmax(one_hot)

# dataset function for folders


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, IMAGE_SIZE)  # [img_height, img_width])


def process_folderpath(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

# Configure dataset for performance


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def show_oneimage_category(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    plt.imshow(image_batch)
    plt.title(CLASSES[label_batch.numpy()])

from tensorflow.keras import layers
def create_model0():
    model = tf.keras.Sequential([
      layers.experimental.preprocessing.Rescaling(1./255),
      layers.Conv2D(32, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dropout(rate=0.5),
      layers.Dense(numclasses, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(lr=0.001) #tf.keras.optimizers.SGD(lr=0.001, momentum=0.01)#optimizers.RMSprop(lr=1e-4)#optimizers.SGD(lr=0.001) #Adam(lr=0.001)

    model.compile(
      optimizer=optimizer, #'adam',
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), #integer label not one-hot encoding
      metrics=['accuracy'])
  ##one-hot encoded, use categorical_crossentropy. Examples (for a 3-class classification): [1,0,0] , [0,1,0], [0,0,1] But if your Yi's are integers, use sparse_categorical_crossentropy. Examples for above 3-class classification problem: [1] , [2], [3]
    return model

def create_model2():
    #pretrained_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False, weights='imagenet')
    #pretrained_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False)
    pretrained_model = tf.keras.applications.Xception(
        input_shape=[*IMAGE_SIZE, 3], include_top=False)
    #pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
    #pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
    #pretrained_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
    # EfficientNet can be loaded through efficientnet.tfkeras library (https://github.com/qubvel/efficientnet)
    #pretrained_model = efficientnet.tfkeras.EfficientNetB0(weights='imagenet', include_top=False)

    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        # tf.keras.layers.Flatten(),
        # the float32 is needed on softmax layer when using mixed precision
        tf.keras.layers.Dense(
            numclasses, activation='softmax', dtype=tf.float32)
    ])

    model.compile(
        optimizer='adam',
        #loss = 'categorical_crossentropy',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_history(history, metric, val_metric):
    acc = history.history[metric]
    val_acc = history.history[val_metric]

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    fig = plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([min(plt.ylim()), 1])
    plt.grid(True)
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title('Training and Validation Loss')
    plt.show()
    fig.savefig('traininghistory.pdf')


def main():
    # set logger
    # log = logger.setup_logger(os.path.join(args.save_path, 'test.log'))
    # for key, value in sorted(vars(args).items()):
    #     log.info(str(key) + ': ' + str(value))

    if args.arch == 'Tensorflow':
        import tensorflow as tf
        print("Tensorflow Version: ", tf.__version__)
        print("Keras Version: ", tf.keras.__version__)
        global AUTOTUNE
        

        if args.GPU:
            physical_devices = tf.config.list_physical_devices('GPU')
            num_gpu = len(physical_devices)
            print("Num GPUs:", num_gpu)
            if num_gpu == 1:
                # default strategy that works on CPU and single GPU
                strategy = tf.distribute.get_strategy()
            else:
                # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines
                strategy = tf.distribute.MirroredStrategy()  # for GPU or multi-GPU machines
            print("Number of accelerators: ", strategy.num_replicas_in_sync)
            # On Colab/GPU, a higher batch size does not help and sometimes does not fit on the GPU (OOM)
            BATCH_SIZE = 32
        elif args.TPU:
            tpu = None
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("Number of accelerators: ", strategy.num_replicas_in_sync)
            # A TPU has 8 cores so this will be 128
            BATCH_SIZE = 16*strategy.num_replicas_in_sync

        AUTOTUNE = tf.data.AUTOTUNE
        if args.data_type == 'TFrecord':
            filenames = tf.io.gfile.glob(args.data_path)
            num_tfrecordfiles = len(filenames)

            split = int(len(filenames) * VALIDATION_SPLIT)
            training_filenames = filenames[split:]
            validation_filenames = filenames[:split]
            print("Pattern matches {} data files. Splitting dataset into {} training files and {} validation files".format(
                len(filenames), len(training_filenames), len(validation_filenames)))
            validation_steps = int(3670 // num_tfrecordfiles *
                                   len(validation_filenames)) // BATCH_SIZE
            steps_per_epoch = int(3670 // num_tfrecordfiles *
                                  len(training_filenames)) // BATCH_SIZE
            print("With a batch size of {}, there will be {} batches per training epoch and {} batch(es) per validation run.".format(
                BATCH_SIZE, steps_per_epoch, validation_steps))

            display_dataset = load_dataset(training_filenames)
            testimage, testlabel = next(iter(display_dataset))
            show_oneimage_category(testimage, testlabel)

            # instantiate the datasets
            train_ds = get_batched_dataset(
                training_filenames, train=True)
            val_ds = get_batched_dataset(
                validation_filenames, train=False)
        elif args.data_type == 'folder':
            data_dir = pathlib.Path(args.data_path)
            filelist = (args.data_path+'/*/*')
            datapattern = glob.glob(filelist)
            image_count = len(datapattern)
            # Option2: list_ds = tf.data.Dataset.list_files(filelist, shuffle=False) # str(data_dir/'*/*'), filelist
            list_ds = tf.data.Dataset.list_files(
                str(data_dir/'*/*'), shuffle=False)
            list_ds = list_ds.shuffle(
                image_count, reshuffle_each_iteration=False)
            global CLASSES
            CLASSES = np.array(sorted(
                [item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
            # Split the dataset into train and validation:
            val_size = int(image_count * VALIDATION_SPLIT)
            train_ds = list_ds.skip(val_size)
            val_ds = list_ds.take(val_size)

            # see the length of each dataset as follows:
            train_len = tf.data.experimental.cardinality(train_ds).numpy()
            print('Training data len:', train_len)
            val_len = tf.data.experimental.cardinality(val_ds).numpy()
            print('Validation data len:', val_len)

            # Use Dataset.map to create a dataset of image, label pairs:
            # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
            train_ds = train_ds.map(
                process_folderpath, num_parallel_calls=AUTOTUNE)
            val_ds = val_ds.map(process_folderpath,
                                num_parallel_calls=AUTOTUNE)
            train_ds = configure_for_performance(train_ds)
            val_ds = configure_for_performance(val_ds)

            image_batch, label_batch = next(iter(train_ds))
            plt.figure(figsize=(10, 10))
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(image_batch[i].numpy().astype("uint8"))
                label = label_batch[i]
                plt.title(CLASSES[label])
                plt.axis("off")

        else:
            print('Data type not correct')

        # batch and learning rate settings
        if strategy.num_replicas_in_sync == 8:  # TPU or 8xGPU
            #BATCH_SIZE = 16 * strategy.num_replicas_in_sync
            #VALIDATION_BATCH_SIZE = 16 * strategy.num_replicas_in_sync
            start_lr = 0.00001
            min_lr = 0.00001
            max_lr = 0.00005 * strategy.num_replicas_in_sync
            rampup_epochs = 5
            sustain_epochs = 0
            exp_decay = .8
        elif strategy.num_replicas_in_sync == 1:  # single GPU
            #BATCH_SIZE = 16
            #VALIDATION_BATCH_SIZE = 16
            start_lr = 0.00001
            min_lr = 0.00001
            max_lr = 0.0002
            rampup_epochs = 5
            sustain_epochs = 0
            exp_decay = .8
        else:  # TPU pod
            #BATCH_SIZE = 8 * strategy.num_replicas_in_sync
            #VALIDATION_BATCH_SIZE = 8 * strategy.num_replicas_in_sync
            start_lr = 0.00001
            min_lr = 0.00001
            max_lr = 0.00002 * strategy.num_replicas_in_sync
            rampup_epochs = 7
            sustain_epochs = 0
            exp_decay = .8

        lr_callback = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: lrfn(epoch), verbose=True)

        rng = [i for i in range(EPOCHS)]
        y = [lrfn(x) for x in rng]
        plt.plot(rng, [lrfn(x) for x in rng])
        print(y[0], y[-1])

        with strategy.scope():  # creating the model in the TPUStrategy scope places the model on the TPU
            model = create_model2()
            #model0 = create_model0()
        model.summary()

        start_time = time.time()
        EPOCHS = 30
        TRAIN_STEPS = 2936 // BATCH_SIZE  # 2936 is the length of train data
        # history = model.fit(train_ds, validation_data=val_ds,
        #                     steps_per_epoch=TRAIN_STEPS, epochs=EPOCHS, callbacks=[lr_callback])
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
        final_accuracy = history.history["val_accuracy"][-5:]
        print("FINAL ACCURACY MEAN-5: ", np.mean(final_accuracy))
        print("TRAINING TIME: ", time.time() - start_time, " sec")

        plot_history(history, 'accuracy', 'val_accuracy')


if __name__ == '__main__':
    main()

# Distributed training with Keras, colab version: https://colab.research.google.com/drive/13vzRVWJFO0rQy9llgLk_tw6d61O0nu1Y#scrollTo=MfBg1C5NB3X0
# Import TensorFlow and TensorFlow Datasets


import configargparse #pip install configargparse
import tensorflow as tf
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import time
import os
print(tf.__version__)

from Datasetutil.TFdatasetutil import loadtfds, loadkerasdataset

model = None 
# import logger

parser = configargparse.ArgParser(description='myTFDistributedClassify')
parser.add_argument('--tfds_dataname', type=str, default='mnist',
                    help='path to get data')
parser.add_argument('--data_path', type=str, default='/home/kaikai/.keras/datasets/flower_photos',
                    help='path to get data')
parser.add_argument('--save_path', type=str, default='./outputs/fashion',
                    help='path to save the model')
parser.add_argument('--data_type', default='folder', choices=['folder', 'TFrecord'],
                    help='the type of data')  # gs://cmpelkk_imagetest/*.tfrec
# network
parser.add_argument('--model_name', default='depth', choices=['disparity', 'depth'],
                    help='the network')
parser.add_argument('--arch', default='Tensorflow', choices=['Tensorflow', 'Pytorch'],
                    help='Model Name, default: Tensorflow.')
parser.add_argument('--batchsize', type=int, default=32,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=12,
                    help='epochs')
parser.add_argument('--GPU', type=bool, default=True,
                    help='use GPU')
parser.add_argument('--TPU', type=bool, default=False,
                    help='use GPU')


args = parser.parse_args()


# def loadtfds(name='mnist'):
#     import tensorflow_datasets as tfds
#     datasets, info = tfds.load(name, with_info=True, as_supervised=True) #downloaded and prepared to /home/lkk/tensorflow_datasets/mnist/3.0.1.
#     train, test = datasets['train'], datasets['test'] 

#     # You can also do info.splits.total_num_examples to get the total
#     # number of examples in the dataset.

#     num_train_examples = info.splits['train'].num_examples
#     num_test_examples = info.splits['test'].num_examples
#     return train, test, num_train_examples, num_test_examples

# Pixel values, which are 0-255, have to be normalized to the 0-1 range. Define this scale in a function.


# def scale(image, label):
#     image = tf.cast(image, tf.float32)
#     image /= 255

#     return image, label


def create_model(strategy,numclasses, metricname='accuracy'):
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(numclasses)
        ])

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[metricname])
        return model

def create_model2():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
    ])

  return model

# Function for decaying the learning rate.
# You can define any decay function you need.
def learningratefn(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))

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
    print("Tensorflow Version: ", tf.__version__)
    print("Keras Version: ", tf.keras.__version__)

    if args.GPU:
        physical_devices = tf.config.list_physical_devices('GPU')
        num_gpu = len(physical_devices)
        print("Num GPUs:", num_gpu)
        # Create a MirroredStrategy object. This will handle distribution, and provides a context manager (tf.distribute.MirroredStrategy.scope) to build your model inside.
        strategy = tf.distribute.MirroredStrategy()  # for GPU or multi-GPU machines
        print("Number of accelerators: ", strategy.num_replicas_in_sync)
        BUFFER_SIZE = 10000
        BATCH_SIZE_PER_REPLICA = args.batchsize #64
        BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


    train_data, test_data, num_train_examples, num_test_examples =loadkerasdataset()
    #train_data, test_data, num_train_examples, num_test_examples = loadtfds(args.tfds_dataname)
    print(num_train_examples)

    # train_data, test_data, num_train_examples, num_test_examples = loadtfds(
    #     args.tfds_dataname)
    # Apply this function to the training and test data, shuffle the training data, and batch it for training.
    # This dataset fills a buffer with buffer_size elements, then randomly samples elements from this buffer, replacing the selected elements with new elements. For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
    # train_ds = train_data.map(scale).cache().shuffle(
    #     BUFFER_SIZE).batch(BATCH_SIZE)
    # val_ds = test_data.map(scale).batch(BATCH_SIZE)
    train_ds = train_data.cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)
    val_ds = test_data.batch(BATCH_SIZE)

    train_dist_dataset = strategy.experimental_distribute_dataset(train_ds)
    test_dist_dataset = strategy.experimental_distribute_dataset(val_ds)

    # Define the checkpoint directory to store the checkpoints
    checkpoint_dir = args.save_path #'./training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    with strategy.scope():
        # Set reduction to `none` so we can do the reduction afterwards and divide by
        # global batch size.
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)
        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)

    with strategy.scope():
        test_loss = tf.keras.metrics.Mean(name='test_loss')

        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')

    global model
    # model, optimizer, and checkpoint must be created under `strategy.scope`.
    with strategy.scope():
        model = create_model2()

        optimizer = tf.keras.optimizers.Adam()

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        
    def train_step(inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_accuracy.update_state(labels, predictions)
        return loss 

    def test_step(inputs):
        images, labels = inputs

        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)
    
    # `run` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                axis=None)

    @tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.run(test_step, args=(dataset_inputs,))
    
    for epoch in range(args.epochs):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches

        # TEST LOOP
        for x in test_dist_dataset:
            distributed_test_step(x)

        if epoch % 2 == 0:
            checkpoint.save(checkpoint_prefix)

        template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                    "Test Accuracy: {}")
        print (template.format(epoch+1, train_loss,
                                train_accuracy.result()*100, test_loss.result(),
                                test_accuracy.result()*100))

        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
    
    #Export the graph and the variables to the platform-agnostic SavedModel format. After your model is saved, you can load it with or without the scope.
    model.save(args.save_path, save_format='tf')

if __name__ == '__main__':
    main()

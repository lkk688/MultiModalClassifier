import configargparse #pip install configargparse
import tensorflow as tf
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import time
import os
print(tf.__version__)

from TFClassifier.Datasetutil.TFdatasetutil import loadTFdataset #loadtfds, loadkerasdataset, loadimagefolderdataset
from TFClassifier.myTFmodels.CNNsimplemodels import createCNNsimplemodel
from TFClassifier.Datasetutil.Visutil import plot25images, plot9imagesfromtfdataset, plot_history
from TFClassifier.myTFmodels.optimizer_factory import build_learning_rate, setupTensorboardWriterforLR

model = None 
# import logger

parser = configargparse.ArgParser(description='myTFDistributedClassify')
parser.add_argument('--data_name', type=str, default='flower',
                    help='data name: mnist, fashionMNIST, flower')
parser.add_argument('--data_type', default='imagefolder', choices=['tfds', 'kerasdataset', 'imagefolder', 'TFrecord'],
                    help='the type of data')  # gs://cmpelkk_imagetest/*.tfrec
parser.add_argument('--data_path', type=str, default='/home/lkk/.keras/datasets/flower_photos',
                    help='path to get data')
parser.add_argument('--img_height', type=int, default=224,
                    help='resize to img height')
parser.add_argument('--img_width', type=int, default=224,
                    help='resize to img width')
parser.add_argument('--save_path', type=str, default='./outputs/',
                    help='path to save the model')
# network
parser.add_argument('--model_name', default='cnnsimple4', choices=['cnnsimple1', 'cnnsimple2', 'cnnsimple3', 'cnnsimple4','mobilenetmodel1'],
                    help='the network')
parser.add_argument('--arch', default='Tensorflow', choices=['Tensorflow', 'Pytorch'],
                    help='Model Name, default: Tensorflow.')
parser.add_argument('--learningratename', default='fixedstep', choices=['fixedstep', 'fixed', 'warmupexpdecay'],
                    help='path to save the model')
parser.add_argument('--batchsize', type=int, default=32,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=15,
                    help='epochs')
parser.add_argument('--GPU', type=bool, default=True,
                    help='use GPU')
parser.add_argument('--TPU', type=bool, default=False,
                    help='use GPU')


args = parser.parse_args()


# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))


def main():
    print("Tensorflow Version: ", tf.__version__)
    print("Keras Version: ", tf.keras.__version__)

    TAG="0630"
    args.save_path=args.save_path+args.data_name+'_'+args.model_name+'_'+TAG
    print("Output path:", args.save_path)

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


    #train_ds, val_ds, class_names, imageshape = loadTFdataset(args.data_name, args.data_type)
    train_ds, val_ds, class_names, imageshape = loadTFdataset(args.data_name, args.data_type, args.data_path, args.img_height, args.img_width, BATCH_SIZE)
    #train_ds, test_data, num_train_examples, num_test_examples, class_names=loadimagefolderdataset('flower')
    #train_data, test_data, num_train_examples, num_test_examples =loadkerasdataset('cifar10')
    #train_data, test_data, num_train_examples, num_test_examples = loadtfds(args.tfds_dataname)

    # train_data, test_data, num_train_examples, num_test_examples = loadtfds(
    #     args.tfds_dataname)

    #Tune for performance
    AUTOTUNE = tf.data.AUTOTUNE #tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    numclasses=len(class_names)

    global model
    metricname='accuracy'
    metrics=[metricname]
    with strategy.scope():
        model = createCNNsimplemodel(args.model_name, numclasses, imageshape, metrics)
    #model = create_model(strategy,numclasses, metricname)
    model.summary()

    # Define the checkpoint directory to store the checkpoints
    checkpoint_dir = args.save_path #'./training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    setupTensorboardWriterforLR(args.save_path)

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=args.save_path),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                        save_weights_only=True),
        build_learning_rate(args.learningratename),
        #tf.keras.callbacks.LearningRateScheduler(learningratefn),
        PrintLR()
    ]

    print("Initial learning rate: ", round(model.optimizer.lr.numpy(), 5))

    #steps_per_epoch = num_train_examples // BATCH_SIZE  # 2936 is the length of train data
    #print("steps_per_epoch:", steps_per_epoch)
    start_time = time.time()
    #train the model 
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    # history = model.fit(train_ds, validation_data=val_ds,
    #                     steps_per_epoch=steps_per_epoch, epochs=EPOCHS, callbacks=[lr_callback])

    valmetricname="val_"+metricname
    final_accuracy = history.history[valmetricname][-5:]
    print("FINAL ACCURACY MEAN-5: ", np.mean(final_accuracy))
    print("TRAINING TIME: ", time.time() - start_time, " sec")

    plot_history(history, metricname, valmetricname, args.save_path)

    #Export the graph and the variables to the platform-agnostic SavedModel format. After your model is saved, you can load it with or without the scope.
    model.save(args.save_path, save_format='tf')

    eval_loss, eval_acc = model.evaluate(val_ds)
    print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

if __name__ == '__main__':
    main()

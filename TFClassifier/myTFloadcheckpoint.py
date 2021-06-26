from __future__ import absolute_import
import configargparse #pip install configargparse
import tensorflow as tf
import TFdatasetutil

# import logger

parser = configargparse.ArgParser(description='myTFDistributedClassify')
parser.add_argument('--tfds_dataname', type=str, default='mnist',
                    help='path to get data')
parser.add_argument('--data_path', type=str, default='/home/kaikai/.keras/datasets/flower_photos',
                    help='path to get data')
parser.add_argument('--checkpoint_path', type=str, default='./logs',
                    help='path to get check point')
parser.add_argument('--save_path', type=str, default='./outputs',
                    help='path to saved model')

args = parser.parse_args()


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

if __name__ == '__main__':
    train_data, test_data, num_train_examples, num_test_examples = TFdatasetutil.loadtfds(args.tfds_dataname)
    print(num_train_examples)

    BUFFER_SIZE = 10000
    BATCH_SIZE =32
    # train_ds = train_data.map(TFdatasetutil.scale).cache().shuffle(
    #     BUFFER_SIZE).batch(BATCH_SIZE)
    val_ds = test_data.map(TFdatasetutil.scale).batch(BATCH_SIZE)

    

    metricname='accuracy'
    numclasses=10
    strategy = tf.distribute.MirroredStrategy()  # for GPU or multi-GPU machines
    model = create_model(strategy,numclasses, metricname)
    model.summary()

    checkpoint_dir='./logs/'
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    eval_loss, eval_acc = model.evaluate(val_ds)

    print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

    model.load_weights('./logs/ckpt_12')
    eval_loss, eval_acc = model.evaluate(val_ds)
    print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

    #tensorboard --logdir=path/to/log-directory

    #Export the graph and the variables to the platform-agnostic SavedModel format. After your model is saved, you can load it with or without the scope.
    model.save(args.save_path, save_format='tf')

    #Load the model without strategy.scope
    unreplicated_model = tf.keras.models.load_model(args.save_path)

    unreplicated_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'])

    eval_loss, eval_acc = unreplicated_model.evaluate(val_ds)

    print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))
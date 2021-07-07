import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#Data augmentation and Dropout layers are inactive at inference time.

def createCNNsimplemodel(name, numclasses, img_shape, metrics=['accuracy']):
    #img_shape (img_height, img_width, 3)
    if name=='cnnsimple1':
        return create_simplemodel1(numclasses, img_shape, metrics)
    elif name=='cnnsimple2':
        return create_simplemodel2(numclasses, img_shape, metrics)
    elif name=='cnnsimple3':
        return create_simplemodel3(numclasses, img_shape, metrics)
    elif name=='cnnsimple4':
        return create_simplemodel4(numclasses, img_shape, metrics)
    elif name=='mobilenetmodel1':
        return create_mobilenetmodel1(numclasses, img_shape, metrics)


def create_simplemodel1(numclasses, img_shape, metrics=['accuracy']):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, 3, activation='relu', input_shape=img_shape), #(28, 28, 1)
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(numclasses)
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=metrics)
    return model

def create_simplemodel2(numclasses, img_shape, metrics=['sparse_categorical_accuracy']):
    """Constructs the ML model used to predict handwritten digits."""

    image = tf.keras.layers.Input(shape=img_shape)#(28, 28, 1))

    y = tf.keras.layers.Conv2D(filters=32,
                                kernel_size=5,
                                padding='same',
                                activation='relu')(image)
    y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same')(y)
    y = tf.keras.layers.Conv2D(filters=32,
                                kernel_size=5,
                                padding='same',
                                activation='relu')(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same')(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(1024, activation='relu')(y)
    y = tf.keras.layers.Dropout(0.4)(y)

    probs = tf.keras.layers.Dense(numclasses, activation='softmax')(y)

    model = tf.keras.models.Model(image, probs, name='simplemodel2')

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.05, decay_steps=100000, decay_rate=0.96)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=metrics) #https://www.tensorflow.org/api_docs/python/tf/keras/metrics/sparse_categorical_accuracy

    return model

def create_simplemodel3(numclasses, img_shape, metrics=['accuracy']):
    model = Sequential([
        #layers.experimental.preprocessing.Rescaling(1./255, input_shape=img_shape),
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=img_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(numclasses) #activation='softmax'
        ])
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=metrics)
    return model

def create_simplemodel4(numclasses, img_shape, metrics=['accuracy']):
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                        input_shape=img_shape),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        data_augmentation,
        #layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(numclasses)
        ])
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=metrics)
    return model

def create_mobilenetmodel1(numclasses, img_shape, metrics=['accuracy']):
    pretrained_model = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
    pretrained_model.trainable = False #True

    model = tf.keras.Sequential([
        pretrained_model,
        #tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(numclasses, activation='softmax')
        # tf.keras.layers.Conv2D(32, 3, activation='relu'),
        # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.GlobalAveragePooling2D(),
        # tf.keras.layers.Dense(numclasses, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        #loss = 'categorical_crossentropy',
        loss = 'sparse_categorical_crossentropy',
        metrics=metrics
    )

    return model


def create_Xceptionmodel1(numclasses, img_shape, metrics=['accuracy']):
    pretrained_model = tf.keras.applications.Xception(input_shape=img_shape, include_top=False, weights='imagenet')
    pretrained_model.trainable = True #False #True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        #tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(numclasses, activation='softmax')
        # tf.keras.layers.Conv2D(32, 3, activation='relu'),
        # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.GlobalAveragePooling2D(),
        # tf.keras.layers.Dense(numclasses, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        #loss = 'categorical_crossentropy',
        loss = 'sparse_categorical_crossentropy',
        metrics=metrics
    )

    return model
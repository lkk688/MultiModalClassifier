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
    elif name=='mobilenetmodel2':
        return create_mobilenetmodel2(numclasses, img_shape, metrics)
    elif name=='xceptionmodel1':
        return create_Xceptionmodel1(numclasses, img_shape, metrics)


def create_simplemodel1(numclasses, img_shape, metrics=['accuracy']):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, 3, activation='relu', input_shape=img_shape), #(28, 28, 1)
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(numclasses)
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),#integer label not one-hot encoding
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

#add data augmentation into the model
def create_mobilenetmodel2(numclasses, img_shape, metrics=['accuracy']):
    #These layers are active only during training, when you call model.fit. They are inactive when the model is used in inference mode in model.evaulate or model.fit.
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])
    
    #tf.keras.applications.MobileNetV2 expects pixel values in [-1, 1], but at this point, the pixel values in your images are in [0, 255]. To rescale them, use the preprocessing method included with the model.
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    #Alternatively, you could rescale pixel values from [0, 255] to [-1, 1] using a Rescaling layer.
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

    # Create the base model from the pre-trained model MobileNet V2
    #IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                                include_top=False,
                                                weights='imagenet')
    #This feature extractor converts each 160x160x3 image into a 5x5x1280 block of features

    base_model.trainable = False
    #When you set layer.trainable = False, the BatchNormalization layer will run in inference mode

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    #average over the spatial 5x5 spatial locations o a single 1280-element vector per image (32, 5, 5, 1280) to (32, 1280)

    prediction_layer = tf.keras.layers.Dense(numclasses, activation='softmax')

    inputs = tf.keras.Input(shape=img_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    #lossfun=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    #optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate/10)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=metrics)

    return model

def create_Xceptionmodel1(numclasses, img_shape, metrics=['accuracy']):
    pretrained_model = tf.keras.applications.Xception(input_shape=img_shape, include_top=False, weights='imagenet')#Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5
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

def create_ResNetmodel1(numclasses, img_shape, metrics=['accuracy']):
    pretrained_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=img_shape) #https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50V2

    # preprocess_input = tf.keras.Sequential([
    #     tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    #     #tf.keras.applications.ResNet50V2.preprocess_input,
    # ])

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    
    pretrained_model.trainable = True #False #True

    header = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        #tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(numclasses, activation='softmax', dtype=tf.float32) # the float32 is needed on softmax layer when using mixed precision
    ])

    inputs = tf.keras.Input(shape=img_shape)
    x = data_augmentation(inputs)
    #x = preprocess_input(x)
    x = pretrained_model(x, training=False)
    outputs = header(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              #loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=metrics)

    return model
import tensorflow as tf
import numpy as np
import argparse
from PIL import Image
import time
import io
from numpy import asarray
#import tflite_runtime.interpreter as tflite
#ref: https://www.tensorflow.org/lite/guide/get_started https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python

def testtfliteexport(saved_model_dir):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

def testtfliteinference(tflite_model_path):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    print(input_details)
    output_details = interpreter.get_output_details()
    print(output_details)

    # Test the model on random input data.
    input_shape = input_details[0]['shape']#[1, 180, 180, 3]

    floating_model = input_details[0]['dtype'] == np.float32

    #image_path='/home/lkk/Developer/MyRepo/MultiModalClassifier/tests/imgdata/sunflower.jpeg'
    image_path='/home/lkk/Developer/MyRepo/MultiModalClassifier/tests/imgdata/rose2.jpeg'
    img_height = input_shape[1] #180
    img_width = input_shape[2] #180
    input_data=loadimage(image_path, img_height, img_width)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)

    classindex = np.argmax(output_data[0], axis=-1)
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    print(class_names[classindex])

def loadimage(path, img_height, img_width):
    # load image
    image = Image.open(path).resize((img_width, img_height))
    image = np.array(image)
    print(np.min(image), np.max(image))#0~255
    input=image[np.newaxis, ...]
    input_data = np.array(input, dtype=np.float32)
    # normalize to the range 0-1
    input_data /= 255.0
    print(np.min(input_data), np.max(input_data)) 
    return input_data


if __name__ == '__main__':
    saved_model_dir = '/home/lkk/Developer/MyRepo/MultiModalClassifier/outputs/flower_xceptionmodel1_0712/'
    #testtfliteexport(saved_model_dir)

    testtfliteinference("converted_model.tflite")

    
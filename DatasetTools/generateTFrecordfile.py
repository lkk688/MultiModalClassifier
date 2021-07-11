import tensorflow as tf
import pathlib
import PIL
import PIL.Image
import numpy as np
import math
import os

class_names = None
IMG_height = 180
IMG_width = 180

def decode_jpeg_and_label(filename):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize the image to the desired size.
    image = tf.image.resize(image, [IMG_height, IMG_width])
    # parse flower name from containing directory
    label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
    label = label.values[-2]
    return image, label

def show_oneimage(image_batch, label_batch):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image_batch)
    plt.title(label_batch.numpy().title())
    fig.savefig('./outputs/plotoneimage.png')#

def resize_and_crop_image(image, label):
    image= tf.image.resize(image, [IMG_height, IMG_width])
    return image, label
    # Resize and crop using "fill" algorithm:
    # always make sure the resulting image
    # is cut out from the source image so that
    # it fills the TARGET_SIZE entirely with no
    # black bars and a preserved aspect ratio.
    # w = tf.shape(image)[0]
    # h = tf.shape(image)[1]
    # print(w, h)
    # tw = TARGET_SIZE[1]
    # th = TARGET_SIZE[0]
    # resize_crit = (w * th) / (h * tw)
    # image = tf.cond(resize_crit < 1,
    #                 lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
    #                 lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
    #                )
    # nw = tf.shape(image)[0]
    # nh = tf.shape(image)[1]
    # image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    # return image, label

def recompress_image(image, label):
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.cast(image, tf.uint8)
    #print("image after cast:", image)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
    return image, label, height, width

# Three types of data can be stored in TFRecords: bytestrings, integers and floats
# They are always stored as lists, a single data element will be a list of size 1
def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats): # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))
  

def to_tfrecord(tfrec_filewriter, img_bytes, label, height, width):  
    class_num = np.argmax(np.array(class_names)==label) # 'roses' => 2 (order defined in CLASSES)
    one_hot_class = np.eye(len(class_names))[class_num]     # [0, 0, 1, 0, 0] for class #2, roses

    feature = {

        "image": _bytestring_feature([img_bytes]), # one image in the list
        "class": _int_feature([class_num]),        # one class in the list
        
        # additional (not very useful) fields to demonstrate TFRecord writing/reading of different types of data
        "label":         _bytestring_feature([label]),          # fixed length (1) list of strings, the text label
        "size":          _int_feature([height, width]),         # fixed length (2) list of ints
        "one_hot_class": _float_feature(one_hot_class.tolist()) # variable length  list of floats, n=len(CLASSES)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def main(Path, PATH_OUTPUT='./outputs/TFrecord/', SHARDS = 16):
    #Path='/home/lkk/.keras/datasets/flower_photos'
    File_pattern = Path+'/*/*.jpg' #GCS_PATTERN = 'gs://flowers-public/*/*.jpg'
    AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API
    #GCS_OUTPUT = 'gs://flowers-public/tfrecords-jpeg-192x192-2/flowers'  # prefix for output file names
    #SHARDS = 16
    TARGET_SIZE = [IMG_height, IMG_width]
    global class_names
    data_dir = pathlib.Path(Path)
    class_names = np.array(sorted(
        [item.name.encode('UTF-8') for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
    #class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'] #[b'daisy', b'dandelion', b'roses', b'sunflowers', b'tulips'] # do not change, maps to the labels in the data (folder names)

    nb_images = len(tf.io.gfile.glob(File_pattern))
    shard_size = math.ceil(1.0 * nb_images / SHARDS) #230
    print("Pattern matches {} images which will be rewritten as {} .tfrec files containing {} images each.".format(nb_images, SHARDS, shard_size))

    filenames = tf.data.Dataset.list_files(File_pattern, seed=35155) # This also shuffles the images
    dataset1 = filenames.map(decode_jpeg_and_label, num_parallel_calls=AUTO)

    image_batch, label_batch = next(iter(dataset1))
    print(image_batch.shape)#(180, 180, 3)
    print(label_batch)#tf.Tensor(b'daisy', shape=(), dtype=string)
    show_oneimage(image_batch, label_batch)

    dataset2 = dataset1.map(resize_and_crop_image, num_parallel_calls=AUTO)  #optional, not useful now

    image_batch, label_batch = next(iter(dataset2))
    show_oneimage(image_batch, label_batch)

    #shard_size = math.ceil(1.0 * nb_images / SHARDS)
    dataset3 = dataset2.map(recompress_image, num_parallel_calls=AUTO)
    dataset3 = dataset3.batch(shard_size) # sharding: there will be one "batch" of images per file 

    image, label, height, width = next(iter(dataset3))

    print("Writing TFRecords")
    #PATH_OUTPUT='./outputs/TFrecord/'
    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)
    for shard, (image, label, height, width) in enumerate(dataset3):
        # batch size used as shard size here
        shard_size = image.numpy().shape[0] #optional
        # good practice to have the number of records in the filename
        filename = PATH_OUTPUT + "{:02d}-{}.tfrec".format(shard, shard_size)
        
        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(shard_size):
                example = to_tfrecord(out_file,
                                        image.numpy()[i], # re-compressed image: already a byte string
                                        label.numpy()[i],
                                        height.numpy()[i],
                                        width.numpy()[i])
                out_file.write(example.SerializeToString())
            print("Wrote file {} containing {} records".format(filename, shard_size))

if __name__ == '__main__':
    Path='/home/lkk/.keras/datasets/flower_photos'
    main(Path)
# MultiModalClassifier
This is a project repo for multi-modal deep learning classifier with popular models from Tensorflow and Pytorch. The goal of these baseline models is to provide a template to build on and can be a starting point for any new ideas, applications. If you want to learn basics of ML and DL, please refer this repo: https://github.com/lkk688/DeepDataMiningLearning.

# Package setup
Install this project in development mode
```bash
(venv38) MyRepo/MultiModalClassifier$ python setup.py develop
```
After the installation, the package "MultimodalClassifier==0.0.1" is installed in your virtual environment. You can check the import
```bash
>>> import TFClassifier
>>> import TFClassifier.Datasetutil
>>> import TFClassifier.Datasetutil.Visutil
```

If you went to uninstall the package, perform the following step
```bash
(venv38) lkk@cmpeengr276-All-Series:~/Developer/MyRepo/MultiModalClassifier$ python setup.py develop --uninstall
```

# Code organization
* [DatasetTools](./DatasetTools): common tools and code scripts for processing datasets
* [TFClassifier](./TFClassifier): Tensorflow-based classifier
  * [myTFDistributedTrainerv2.py](./TFClassifier/myTFDistributedTrainerv2.py): main training code
  * [myTFInference.py](./TFClassifier/myTFInference.py): main inference code
  * [exportTFlite.py](./TFClassifier/exportTFlite.py): convert form TF model to TFlite
* [TorchClassifier](./TorchClassifier): Pytorch-based classifier
  * [myTorchTrainer.py](./TorchClassifier/myTorchTrainer.py): Pytorch main training code
  * [myTorchEvaluator.py](./TorchClassifier/myTorchEvaluator.py): Pytorch model evaluation code 

# PytorchClassifier (New)
## Test CIFAR10:
```bash
python myTorchTrainer.py --data_name 'CIFAR10' --data_type 'torchvisiondataset' --data_path r"E:\Dataset" --model_name 'cnnmodel1' --learningratename 'ConstantLR' --optimizer 'SGD'
```

## ImageNet Training
Download ImageNet dataset (tiny-imagenet-200, ImageNet-Blur, imagenet21k_resized.tar.gz) from https://image-net.org. 

Test train based on tiny-imagenet-200 dataset

```bash
(mycondapy310) [010796032@g4 TorchClassifier]$ python myTorchTrainer.py --data_name 'tiny-imagenet-200' --data_type 'trainonly' --data_path "/data/cmpe249-fa23/ImageClassData" --model_name 'resnetmodel1' --learningratename 'StepLR' --lr 0.1 --momentum 0.9 --wd 1e-4 --optimizer 'SGD'
```
Trained model in "TorchClassifier/outputs/tiny-imagenet-200_resnetmodel1_0910"

```bash
(mycondapy310) [010796032@g4 TorchClassifier]$ python myTorchTrainer.py --data_name 'tiny-imagenet-200' --data_type 'trainonly' --data_path "/data/cmpe249-fa23/ImageClassData" --model_name 'vit_b_32' --learningratename 'StepLR' --lr 0.1 --momentum 0.9 --wd 1e-4 --optimizer 'SGD' --batchsize 32
```

```bash
(mycondapy310) [010796032@g4 TorchClassifier]$ python myTorchTrainer.py --data_name 'tiny-imagenet-200' --data_type 'trainonly' --data_path "/data/cmpe249-fa23/ImageClassData" --model_name 'vit_b_16' --learningratename 'StepLR' --lr 0.1 --momentum 0.9 --wd 1e-4 --optimizer 'Adam' --batchsize 32 --TAG '0910'
```

```bash
(mycondapy310) [010796032@g4 TorchClassifier]$ python myTorchTrainer.py --data_name 'flower_photos' --data_type 'traintestfolder' --data_path "/data/cmpe249-fa23/ImageClassData" --model_name 'vit_b_16' --learningratename 'StepLR' --lr 0.1 --momentum 0.9 --wd 1e-4 --optimizer 'Adam' --batchsize 64 --TAG '0916'
```

Complete the ImageNet-Blur training in HPC2, trained model saved in "outputs/imagenet_blurred_resnet50_0328"
```bash
python myTorchTrainer.py --data_name 'imagenet_blurred' --data_type 'trainonly' --data_path "/data/cmpe249-fa23/ImageClassData" --model_name 'resnet50' --learningratename 'StepLR' --lr 0.1 --momentum 0.9 --wd 1e-4 --optimizer 'SGD'

...
Test Accuracy (Overall): 61% (158825/256213)

```

# ImageNet Evaluation

Tiny Imagenet evaluation
```bash
  $ python myTorchEvaluator.py --data_name 'tiny-imagenet-200' --data_type 'trainonly' --data_path "/data/cmpe249-fa23/ImageClassData" --model_name 'resnet50' --checkpoint 'outputs/tiny-imagenet-200_resnet50_0328/checkpoint.pth.tar' --classmap 'TorchClassifier/Datasetutil/tinyimagenet_idmap.json' --gpuid 0
```
Imagenet_blurred evaluation
```bash
  $ python myTorchEvaluator.py --data_name 'imagenet_blurred' --data_type 'trainonly' --data_path "/data/cmpe249-fa23/ImageClassData" --model_name 'resnet50' --model_type 'ImageNet' --classmap 'TorchClassifier/Datasetutil/imagenet1000id2label.json' --gpuid 0
#image_path="/data/cmpe249-fa22/ImageClassData/tiny-imagenet-200/train/n04285008/images/n04285008_31.JPEG"

```

```bash
  $ python myTorchEvaluator.py --data_name 'imagenet_blurred' --data_type 'trainonly' --data_path "/data/cmpe249-fa23/ImageClassData" --model_name 'vit_b_32' --model_type 'ImageNet' --classmap 'TorchClassifier/Datasetutil/imagenet1000id2label.json' --gpuid 0
#image_path="/data/cmpe249-fa22/ImageClassData/tiny-imagenet-200/train/n04285008/images/n04285008_31.JPEG"

```

```bash
(mycondapy310) [010796032@g4 TorchClassifier]$ python myTorchEvaluator.py --data_name 'imagenet_blurred' --data_type 'trainonly' --data_path "/data/cmpe249-fa23/ImageClassData" --model_name 'vit_b_32' --model_type 'ImageNet' --classmap 'TorchClassifier/Datasetutil/imagenet1000id2label.json' --gpuid 0

Test Accuracy (Overall): 77% (198655/256213)
```

#image_path="/data/cmpe249-fa23/ImageClassData/tiny-imagenet-200/train/n04285008/images/n04285008_31.JPEG"


# Tensorflow Lite
* Tensorflow lite guide [link](https://www.tensorflow.org/lite/guide)
* [exportTFlite](\TFClassifier\exportTFlite.py) file exports model to TFlite format.
  * testtfliteexport function exports the float format TFlite model
  * tflitequanexport function exports the TFlite model with post-training quantization, the model size can be reduced by
![image](https://user-images.githubusercontent.com/6676586/126202680-e2e53942-7951-418c-a461-99fd88d2c33e.png)
  * The converted quantized model won't be compatible with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU) because the input and output still remain float in order to have the same interface as the original float only model.
* To ensure compatibility with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU), we can enforce full integer quantization for all ops including the input and output, add the following code into function tflitequanintexport
```bash
converter_int8.inference_input_type = tf.int8  # or tf.uint8
converter_int8.inference_output_type = tf.int8  # or tf.uint8
```
  * The check of the floating model during inference will show false
```bash
floating_model = input_details[0]['dtype'] == np.float32
```
  * When preparing the image data for the int8 model, we need to conver the uint8 (0-255) image data to int8 (-128-127) via loadimageint function
  
# TensorRT inference
Check this [Colab](https://colab.research.google.com/drive/1aCbuLCWEuEpTVFDxA20xKPFW75FiZgK-?usp=sharing) (require SJSU google account) link to learn TensorRT inference for Tensorflow models.
Check these links for TensorRT inference for Pytorch models: 
* https://github.com/NVIDIA-AI-IOT/torch2trt
* https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
* https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt/

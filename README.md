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

## ImageNet
Download ImageNet dataset (tiny-imagenet-200, ImageNet-Blur, imagenet21k_resized.tar.gz) from https://image-net.org. 

Test train based on tiny-imagenet-200 dataset

```bash
python myTorchTrainer.py --data_name 'tiny-imagenet-200' --data_type 'trainonly' --data_path r"E:\Dataset\ImageNet\tiny-imagenet-200" --model_name 'resnetmodel1' --learningratename 'StepLR' --lr 0.1 --momentum 0.9 --wd 1e-4 --optimizer 'SGD'

Epoch 39/39
----------
train Loss: 0.9043 Acc: 0.7769
val Loss: 2.1729 Acc: 0.5147

Training complete in 133m 43s
Best val Acc: 0.517000
Test Loss: 0.437732
...

Test Accuracy of n09193705: 55% (55/100)
Test Accuracy of n09246464: 42% (43/101)
Test Accuracy of n09256479: 58% (64/110)
Test Accuracy of n09332890: 37% (42/113)
Test Accuracy of n09428293: 47% (43/91)
Test Accuracy of n12267677: 47% (47/100)

Test Accuracy (Overall): 51% (10218/20000)
```

Complete the ImageNet-Blur training in HPC2, trained model saved in "outputs/imagenet_blurred_resnet50_0328"
```bash
python myTorchTrainer.py --data_name 'imagenet_blurred' --data_type 'trainonly' --data_path "/data/cmpe249-fa22/ImageClassData" --model_name 'resnet50' --learningratename 'StepLR' --lr 0.1 --momentum 0.9 --wd 1e-4 --optimizer 'SGD'

...
Test Accuracy (Overall): 61% (158825/256213)

```


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

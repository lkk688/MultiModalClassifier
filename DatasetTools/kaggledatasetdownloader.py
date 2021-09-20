#pip install kaggle
#copy the kaggle.json to ~/.kaggle folder
#chmod 600 /home/lkk/.kaggle/kaggle.json
#Download birds dataset: kaggle datasets download veeralakrishna/200-bird-species-with-11788-images -p ~/Developer/MyRepo/ImageClassificationData/ --unzip
#Downloading 200-bird-species-with-11788-images.zip to /home/lkk/Developer/MyRepo/ImageClassificationData
import torchvision.datasets as datasets
import glob
import os
import shutil
import torch
from torchvision import transforms

ROOT='/home/lkk/Developer/MyRepo/ImageClassificationData' #CUB_200_2011.tgz under this folder
datapath=os.path.join(ROOT,'CUB_200_2011.tgz')
#datasets.utils.extract_archive(datapath, ROOT)#created folder /CUB_200_2011

datapathpatern='/home/lkk/Developer/MyRepo/ImageClassificationData'+'/CUB_200_2011/*'
print(glob.glob(datapathpatern))

TRAIN_RATIO = 0.8

data_dir = os.path.join(ROOT, 'CUB_200_2011')
images_dir = os.path.join(data_dir, 'images')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

if os.path.exists(train_dir):
    shutil.rmtree(train_dir) 
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

os.makedirs(train_dir)
os.makedirs(test_dir)

classes = os.listdir(images_dir)

for c in classes:
    print("Class:", c)
    class_dir = os.path.join(images_dir, c)
    print("class_dir:", class_dir)
    
    images = os.listdir(class_dir)
       
    n_train = int(len(images) * TRAIN_RATIO)
    print("number of training images:", n_train)
    
    train_images = images[:n_train]
    test_images = images[n_train:]
    print("number of test images:", len(test_images))
    
    os.makedirs(os.path.join(train_dir, c), exist_ok = True)
    os.makedirs(os.path.join(test_dir, c), exist_ok = True)
    
    for image in train_images:
        image_src = os.path.join(class_dir, image)
        image_dst = os.path.join(train_dir, c, image) 
        shutil.copyfile(image_src, image_dst)
        
    for image in test_images:
        image_src = os.path.join(class_dir, image)
        image_dst = os.path.join(test_dir, c, image) 
        shutil.copyfile(image_src, image_dst)

train_data = datasets.ImageFolder(root = train_dir, 
                                  transform = transforms.ToTensor())

means = torch.zeros(3)
stds = torch.zeros(3)

for img, label in train_data:
    means += torch.mean(img, dim = (1,2))
    stds += torch.std(img, dim = (1,2))

means /= len(train_data)
stds /= len(train_data)
    
print(f'Calculated means: {means}')#[0.4872, 0.5005, 0.4325]
print(f'Calculated stds: {stds}')#[0.1822, 0.1811, 0.1931]
#final datapath: /home/lkk/Developer/MyRepo/ImageClassificationData/CUB_200_2011/train, test
import os
import json
import glob
import torch
from torchvision.io import read_image
from torchvision import transforms
from albumentations import Resize, Compose #pip install albumentations #Fast image augmentation library
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import Normalize
import cv2
from PIL import Image

#ImageNet means and stds
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]

def preprocess_image(img_path, imagesize=224):
    #img = read_image(img_path)# [3, 64, 64]
    img = Image.open(img_path)
    datatransform = transforms.Compose([
                    transforms.Resize(imagesize, interpolation=3),
                    transforms.CenterCrop(imagesize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = pretrained_means, std = pretrained_stds)
                                     ])
    # Step 3: Apply inference preprocessing transforms
    batch_data = datatransform(img)[None,]
    # batch_data = datatransform(img.numpy())
    # batch_data = batch_data.unsqueeze(0)
    return batch_data


def preprocess_imagecv2(img_path, imagesize=224):
    # transformations for the input data
    transforms = Compose([
        Resize(imagesize, imagesize, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=pretrained_means, std=pretrained_stds),
        ToTensorV2(),
    ])
    
    # read input image
    input_img = cv2.imread(img_path)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)#convert it to the RGB colorspace
    
    # do transformations
    input_data = transforms(image=input_img)["image"] #[3, 224, 224]
    batch_data = torch.unsqueeze(input_data, 0) #[1, 3, 224, 224]
    return batch_data

def writedicttojson(Folder, dict, filename="imagenet_idmap.json"):
    with open(os.path.join(Folder, filename), "w") as file:
        file.write(json.dumps(dict))

#https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/classify.py
def load_jpeg_from_file(path, image_size, cuda=True):
    img_transforms = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    img = img_transforms(Image.open(path))
    with torch.no_grad():
        # mean and std are not multiplied by 255 as they are in training script
        # torch dataloader reads data into bytes whereas loading directly
        # through PIL creates a tensor with floats in [0,1] range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if cuda:
            mean = mean.cuda()
            std = std.cuda()
            img = img.cuda()
        img = img.float()

        input = img.unsqueeze(0).sub_(mean).div_(std)

    return input

#"n02124075": "Egyptian cat"
def loadjsontodict(Path):
    if Path and os.path.isfile(Path):
        f = open(Path)
        data = json.load(f)
        return data
    else:
        return {}

import ast
def loaddictfromtxt(Path):
    if Path and os.path.isfile(Path):
        with open(Path) as f:
            data = f.read()
        print("Data type before reconstruction : ", type(data))
        # reconstructing the data as a dictionary
        js = ast.literal_eval(data)
        #js = json.loads(data)
        print("Data type after reconstruction : ", type(js))
        print(js)
        return js
    else:
        return {}

def dict2array(dicts):
    varray=[]
    for k, v in dicts.items():
        varray.append(v)
    #test=[v for k, v in dicts]
    return varray

def foldernames2idmap(Folder):
    folderimage_map = {}
    dirs = glob.glob(Folder + "*")
    for i, dir in enumerate(dirs):
        tmp=dir.split("/")
        name=tmp[-1]
        folderimage_map[name] = i
    print(len(folderimage_map))
    return folderimage_map



if __name__ == "__main__":
    #https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/imagenet_info.py
    # to_label = None
    # label_type = 'name'
    # if label_type in ('name', 'description', 'detail'):
    #     imagenet_subset = infer_imagenet_subset(model)
    #     if imagenet_subset is not None:
    #         dataset_info = ImageNetInfo(imagenet_subset)
    #         if args.label_type == 'name':
    #             to_label = lambda x: dataset_info.index_to_label_name(x)
    #         elif args.label_type == 'detail':
    #             to_label = lambda x: dataset_info.index_to_description(x, detailed=True)
    #         else:
    #             to_label = lambda x: dataset_info.index_to_description(x)
    #         to_label = np.vectorize(to_label)
    #     else:
    #         _logger.error("Cannot deduce ImageNet subset from model, no labelling will be performed.")

    dict=loaddictfromtxt("TorchClassifier/Datasetutil/imagenet1000idx2label.txt")
    writedicttojson("TorchClassifier/Datasetutil/", dict, "imagenet1000id2label.json")
    
    #check folder names
    Folder = "/data/cmpe249-fa22/ImageClassData/tiny-imagenet-200/train/"
    folderimage_map=foldernames2idmap(Folder) #200

    Folder="/data/cmpe249-fa22/ImageClassData/imagenet_blurred/train/"
    folderimage_map=foldernames2idmap(Folder) #1000

    #tiny-imagenet
    ALL_IDS = "/data/cmpe249-fa22/ImageClassData/tiny-imagenet-200/wnids.txt" #200 ids
    sub_ids_map = {}
    with open(ALL_IDS, "rb") as allids_file:
        rows = allids_file.readlines()
        for row in rows:
            row = row.strip().decode("utf-8")
            sub_ids_map[row] = 0


    MAP_ID2WORD = "/data/cmpe249-fa22/ImageClassData/tiny-imagenet-200/words.txt"
    id_word_map = {}
    with open(MAP_ID2WORD, "rb") as map_class_file:
        rows = map_class_file.readlines()
        for row in rows:
            row = row.strip()
            arr = row.decode("utf-8").split("\t")
            key = arr[0]
            id_word_map[key] = arr[1]
            if key in sub_ids_map.keys():
                sub_ids_map[key] = arr[1]


    print(len(id_word_map)) #82115
    print(len(sub_ids_map))

    currentfolder=os.getcwd()
    writedicttojson(currentfolder, sub_ids_map, "tinyimagenet_idmap.json")
    writedicttojson(currentfolder, id_word_map, "imagenet_idmap.json")
    #check saved json file
    tinyids=loadjsontodict("tinyimagenet_idmap.json")
    fullids=loadjsontodict("imagenet_idmap.json")

    # with open(os.path.join(currentfolder, "tinyimagenet_idmap.json"), "w") as file:
    #     file.write(json.dumps(sub_ids_map))

    # with open(os.path.join(currentfolder, "imagenet_idmap.json"), "w") as file:
    #     file.write(json.dumps(id_word_map))

    
import torch
import pandas as pd
from torchvision import datasets, models, transforms
import os
import glob
from PIL import Image
import numpy as np
import cv2

# basePath = '/content/'
# data_dir = os.path.join(basePath, 'Output_Images')

# input_size = 224
# batch_size = 16

# for split_dir in ['train', 'val']:
#     split_path = os.path.join(data_dir, split_dir)
#     for cat in os.listdir(split_path):     
#         for img_file in os.listdir(os.path.join(split_path, cat)):

#             # print(os.path.join(split_path, img_file))   
#             img = cv2.imread(os.path.join(split_path, cat, img_file), cv2.IMREAD_COLOR)
#             print(img.shape)

# image_types = ["Covid-19","No_findings","Pneumonia"]
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device

# def resize_images(image_dir):
#   for im_type in image_types:
#     # Iterate through each image file in each image_type folder
#     #  glob reads in any image with the extension "image_dir/im_type/*"
#     for file in glob.glob(os.path.join(image_dir, im_type, "*")):
#         im = Image.open(file)
#         f, e = os.path.splitext(file)
#         imResize = im.resize((input_size,input_size), Image.ANTIALIAS)
#         os.remove(file)
#         imResize.save(f + '.png', 'PNG', quality=90)

# !cp -R '/content/drive/My Drive/COVID-19/Output_Images' '/content/'
#!cp -R '/content/drive/My Drive/VR/AVR/COVID_project/Sample' '/content/'
# resize_images(data_dir+'/train')
# resize_images(data_dir+'/val')

# def getTrainDataLoaders():
#     # Data augmentation and normalization for training
#     # Just normalization for validation
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#     }

#     print("Initializing Datasets and Dataloaders...")

#     # Create training and validation datasets
#     image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
#     # Create training and validation dataloaders
#     dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

#     return dataloaders_dict

def getModels(device):
    mobilenet = models.mobilenet_v2(pretrained=True)
    mobilenet = mobilenet.to(device)

    squeezenet = models.squeezenet1_1(pretrained=True)
    squeezenet = squeezenet.to(device)
    return mobilenet, squeezenet

def get_mob_squ_features(batch_stacked_images, batch_size):
    batch_stacked_images = np.reshape(batch_stacked_images, (batch_size, 3, batch_stacked_images.shape[0], batch_stacked_images.shape[1]))
    # batch_stacked_images = np.transpose([3, 2, 0, 1])
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)
    model1, model2 = getModels(device) 
    model1.eval()
    model2.eval()
    
    features1 = np.asarray([])
    features2 = np.asarray([])
    feature_labels = []

    print("Features generation started...")
    inputs = torch.from_numpy(batch_stacked_images).float()
    # inputs = inputs.type(torch.DoubleTensor)
    inputs.to(device)
    # inputs.cuda()
    print(inputs.shape)
    # shape = torch.Size((8,3,224,224))
    # x = torch.cuda.FloatTensor(batch_stacked_images.shape)
    # inputs = torch.cuda.FloatTensor(inputs)
    # print(type(inputs), inputs.shape)
    # inputs = torch.randn(shape, out=batch_stacked_images)
    # inputs.cuda()  
    # print("hi", inputs.shape)
    outputs = model1(inputs)
    cpu_outputs = outputs.cpu().detach().numpy()
    mob_features = cpu_outputs

    outputs = model2(inputs)
    cpu_outputs = outputs.cpu().detach().numpy()
    squ_features = cpu_outputs
    print("Feature generation done...")
    print("mob_features:", mob_features.shape, "squ_features:", squ_features.shape)
    return mob_features, squ_features

# def get_mob_squ_features(stacked_images, )

# mobFeatures,squFeatures = get_mob_squ_features(mobilenet,squeezenet)
# print(mobFeatures.shape,squFeatures.shape,labels.shape)

# print(mobFeatures.shape,squFeatures.shape,labels.shape)

# np.reshape(labels, (768, 1)).shape

# mob_feats_with_labels = np.concatenate((mobFeatures, np.reshape(labels, (768, 1))), axis=1)
# print(mob_feats_with_labels.shape)

# squ_feats_with_labels = np.concatenate((squFeatures, np.reshape(labels, (768, 1))), axis=1)
# print(squ_feats_with_labels.shape)

# df_mob = pd.DataFrame(data = mob_feats_with_labels) 
# df_mob.columns = pd.RangeIndex(1, len(df_mob.columns)+1)

# df_mob

# df_squ = pd.DataFrame(data = squ_feats_with_labels) 
# df_squ.columns = pd.RangeIndex(1, len(df_squ.columns)+1)

# df_squ

# df_mob.to_csv('/content/feat_mob.csv')
# df_squ.to_csv('/content/feat_squ.csv')

# !cp '/content/feat_mob.csv' '/content/drive/My Drive/COVID-19'

# !cp '/content/feat_squ.csv' '/content/drive/My Drive/COVID-19'

# np.savetxt('/content/merged_feats.csv', mob_feats_with_labels, delimiter=',')

# np.savetxt('/content/mobilenet_feats.csv', mob_feats_with_labels, delimiter=',')

# np.savetxt('/content/squeezenet_feats.csv', squ_feats_with_labels, delimiter=',')


# from rough import *
from fuzzy_logic import *
from collect_features import *
from stacking import *
from genetic_algo import *
import cv2
import numpy as np
import os

# def fun(a=2):
#     def fun1():
#         return a
#     return fun1()

def genetic_svm_model(images, bs, resource_dir = '/content/drive/My Drive/COVID-19/smo_resources'):
    # path1 = '/media/mkd/New Volume/Documents/Advanced Visual Recognition/COVID Detection/Models/COVID_19/smo/x_ray_sample/train/Covid-19/old#covid-19-pneumonia-7-PA.jpg'
    # arr = np.empty((220,224,0), order='F')
    # img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    # print(img1.shape)
    # img1 = cv2.resize(img1, (224, 220))
    # path2 = '/media/mkd/New Volume/Documents/Advanced Visual Recognition/COVID Detection/Models/COVID_19/smo/x_ray_sample/train/Covid-19/old#covid-19-pneumonia-24-day6.jpg'
    # img2 = cv2.imread(path2, cv2.IMREAD_COLOR)
    # img2 = cv2.resize(img2, (224, 220))
    # print(img2.shape)
    # img = np.concatenate((arr, img1), axis=2)
    # img = np.concatenate((img, img2), axis=2)
    # # # img1 = img[:, :, :3]
    # print(img.shape)
    # # # print(img.shape, img[:, :, :, 0].shape)   
    # bs = 2
    # img = np.reshape(img, (img.shape[0], img.shape[1], 3, bs), order='F')
    # # print(img.shape)
    # # # img1 = cv2.cvtColor(img[:, :, :, 0], cv2.COLOR_BGR2RGB)
    # cv2.imwrite('/media/mkd/New Volume/Documents/Advanced Visual Recognition/COVID Detection/Models/COVID_19/smo/x_ray_sample/train/Covid-19/a.jpg',img[:, :, :, 1]) 
    # print(fun())
    # data_path = '/content/drive/My Drive/COVID-19/x_ray_sample/train/Covid-19'
    # images = np.empty((224, 224, 0), order='F')
    # for file_name in os.listdir(data_path):
    #     img_file = os.path.join(data_path, file_name)
    #     img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    #     img = cv2.resize(img, (224, 224))
    #     images = np.concatenate((images, img), axis=2)
    # bs = 6
    images_copy = np.copy(images, order='F')
    print(images.shape, type(images), images_copy.shape, type(images_copy))
    fuzzy_images = get_fuzzy_images(images, bs)
    print(fuzzy_images.shape)
    stacked_images = get_stacked_images(images_copy, fuzzy_images, bs)
    print(stacked_images.shape)
    mob_features, squ_features = get_mob_squ_features(stacked_images, bs)    
    # resource_dir = '/content/drive/My Drive/COVID-19/smo_resources'
    pred_prob = get_predictions(resource_dir, mob_features, squ_features)
    print(pred_prob.shape)
    return pred_prob    
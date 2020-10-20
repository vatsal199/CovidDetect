import cv2 as cv
import numpy as np
import math
import time
from matplotlib import pyplot as plt
# from google.colab.patches import cv2_imshow

n = 2 # number of rows (windows)
m = 2 # number of colomns (windows)
GAMMA = 1
EPSILON = 0.00001
IDEAL_VARIANCE = 0.35
x0 = x1 = y0 = y1 = WIDTH = HEIGHT = 0

"""## Define Constants:"""

# n = 2 # number of rows (windows on columns)
# m = 2 # number of colomns (windows on rows)
# EPSILON = 0.00001
# #GAMMA, IDEAL_VARIANCE 'maybe' have to changed from image to another 
# GAMMA = 1 # Big GAMMA >> Big mean >> More Brightness
# IDEAL_VARIANCE = 0.35 #Big value >> Big variance >> Big lamda >> more contrast

"""## Call your image:"""

# img_name = '/content/drive/My Drive/VR/AVR/COVID_project/Dataset_2/train/Pneumonia/00003577_001.png'
# img = cv.imread('i:/images/'+img_name)
# img_real = cv.imread('/content/drive/My Drive/VR/AVR/COVID_project/Dataset_2/train/Pneumonia/00003577_001.png')
# img = cv.resize(img_real, (224, 224))
# layer = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# WIDTH = layer.shape[1]
# HEIGHT = layer.shape[0]
# x0, x1, y0, y1 = 0, WIDTH - 1, 0, HEIGHT - 1

"""## Define the essential functions:
- phy: E --> R
- multiply: ExE --> R
- norm: E --> R+
- scalar_multiply: The scalar is defined on R, The result value is on E
- addition: E+E --> E
- subtract: E-E --> E
- C: defined on R
- qx: [x0, x1] --> [0, 1]
- qy: [y0, y1] --> [0, 1]
- p: [x0, x1]x[y0, y1] --> [0, 1]
- w: [x0, x1]x[y0, y1] --> [0, 1]
- mapping: map values from range to range
"""

# split the image to windows
def phy(value): # phy: E --> R 
    #if ((1+value)/((1-value)+0.0001)) < 0:
    #print(value)
    return 0.5 * np.log((1+value)/((1-value)+EPSILON))

def multiplication(value1, value2): # ExE --> R
    return phy(value1) * phy(value2)

def norm(value):
    return abs(phy(value))

def scalar_multiplication(scalar, value):# value in E ([-1,1])
    s = (1+value)**scalar
    z = (1-value)**scalar
    res = (s-z)/(s+z+EPSILON)
    return res

def addition(value1, value2): # value1,value2 are in E ([-1,1])
    res = (value1+value2)/(1+(value1*value2)+EPSILON)
    return res

def subtract(value1, value2): # value1,value2 are in E ([-1,1])
    res = (value1-value2)/(1-(value1*value2)+EPSILON)
    return res

def C(m,i):
    return math.factorial(m)/((math.factorial(i)*math.factorial(m-i))+EPSILON)

def qx(i, x): # i: window index in rows, x: number of current pixel on x-axis
    if (x == WIDTH - 1):
        return 0
    return C(m,i)*(np.power((x-x0)/(x1-x), i) * np.power((x1-x)/(x1-x0), m)) #This is the seconf implementation
    #return C(m,i)*((np.power(x-x0,i) * np.power(x1-x,m-i)) / (np.power(x1-x0,m)+EPSILON))

def qy(j, y):
    '''
    The second implementation for the formula does not go into overflow.
    '''
    if (y == HEIGHT - 1):
        return 0
    return C(n,j)*(np.power((y-y0)/(y1-y), j) * np.power((y1-y)/(y1-y0), n)) #This is the seconf implementation
    #return C(n,j)*((np.power((y-y0),j) * np.power((y1-y),n-j))/ (np.power(y1-y0,n)+EPSILON))
    
def p(i, j, x, y):
    return qx(i, x) * qy(j, y)

def mapping(img, source, dest):
    return (dest[1] - dest[0])*((img - source[0]) / (source[1] - source[0])) + dest[0]

"""## The enhacement phases are:
- Image Fuzzification.
- Calculate Ps and Ws.
- Calculate cards, means, variances, lamdas.
- Implement windows enhacement.
- Implement Image Enhacement.

### 1- Image Fuzzification:
here we convert the image from [0, 255] to [-1, 1]:
"""

# e_layer_gray = mapping(layer, (0, 255), (-1, 1))

"""### 2- Calculate Ps and Ws:"""

def cal_ps_ws(m, n, w, h, gamma):
    ps = np.zeros((m, n, w, h))
    for i in range(m):
        for j in range(n):
            for k in range(w):
                for l in range(h):    
                    ps[i, j, k, l] = p(i, j, k, l)

    ws = np.zeros((m, n, w, h))
    for i in range(m):
        for j in range(n):
            ps_power_gamma = np.power(ps[i, j], gamma)
            for k in range(w):
                for l in range(h):    
                    ws[i, j, k, l] = ps_power_gamma[k, l] / (np.sum(ps[:, :, k, l])+EPSILON)
    return ps, ws
# print('Ps and Ws calculation is in progress...')
# start = time.time()
# ps, ws = cal_ps_ws(m, n, WIDTH, HEIGHT, GAMMA)
# end = time.time()
# print('Ps and Ws calculation has completed successfully in '+str(end-start)+' s')

"""### 3- Calculate cards, means, variances, lamdas:
for each window we have card, mean, variance, lamda to all pixels:
"""

def cal_means_variances_lamdas(w, e_layer):
    means = np.zeros((m, n))
    variances = np.zeros((m, n))
    lamdas = np.zeros((m, n))
    taos = np.zeros((m, n))
    def window_card(w):
        return np.sum(w)

    def window_mean(w, i, j):
        mean = 0
        for k in range(HEIGHT):
            for l in range(WIDTH):
                mean = addition(mean, scalar_multiplication(w[i, j, l, k], e_layer[k, l]))
        mean /= window_card(w[i, j])
        return mean

    def window_variance(w, i, j):
        variance = 0
        for k in range(HEIGHT):
            for l in range(WIDTH):
                variance += w[i, j, l, k] * np.power(norm(subtract(e_layer[k, l], means[i, j])), 2)
        variance /= window_card(w[i, j])
        return variance

    def window_lamda(w, i, j):
        return np.sqrt(IDEAL_VARIANCE) / (np.sqrt(variances[i, j])+EPSILON)

    def window_tao(w, i, j):
        return window_mean(w, i, j)

    for i in range(m):
        for j in range(n):
            means[i, j] = window_mean(w, i, j)
            variances[i, j] = window_variance(w, i, j)
            lamdas[i, j] = window_lamda(w, i, j)
    taos = means.copy()
    
    return means, variances, lamdas, taos
# print('means, variances, lamdas and taos calculation is in progress...')
# start = time.time()
# means, variances, lamdas, taos = cal_means_variances_lamdas(ws, e_layer_gray)
# end = time.time()
# print('means, variances, lamdas and taos calculation is finished in ' + str(end-start) + ' s')

"""### 4- Implement window enhacement:"""

def window_enh(w, i, j, e_layer, lamdas, taos):
    return scalar_multiplication(lamdas[i, j], subtract(e_layer, taos[i, j]))

"""### 5- Implement Image Enhacement:"""

def image_enh(w, e_layer, lamdas, taos):
    new_image = np.zeros(e_layer.shape)
    width = e_layer.shape[1]
    height = e_layer.shape[0]
    for i in range(m):
        for j in range(n):
            win = window_enh(w, i, j, e_layer, lamdas, taos)
            w1 = w[i, j].T.copy()
            for k in range(width):
                for l in range(height):
                    new_image[l, k] = addition(new_image[l, k], scalar_multiplication(w1[l, k], win[l, k]))
    return new_image

"""One layer enhacement function:"""

def one_layer_enhacement(ws, e_layer, lamdas, taos):
    #card_image = layer.shape[0]*layer.shape[1]
    new_E_image = image_enh(ws, e_layer, lamdas, taos)
    res_image = mapping(new_E_image, (-1, 1), (0, 255))
    res_image = np.round(res_image)
    res_image = res_image.astype(np.uint8)
    return res_image

# """## Implement Fuzzy Enhacement:"""

# res_img = one_layer_enhacement(e_layer_gray)

# """## Show the result:"""

# plt.subplot(1, 2, 1)
# plt.imshow(img, cmap = 'gray')
# plt.subplot(1, 2, 2)
# plt.imshow(res_img, cmap = 'gray')
# plt.title('Fuzzy Grayscale image enhacement.')
# plt.show()

# """#Doing It For All RGB Images"""

# import glob
# import os

# image_dir_training = "/content/drive/My Drive/VR/AVR/COVID_project/Dataset_2/test"
# image_types = ["Covid-19", "No_findings","Pneumonia"]
#Constants

# for im_type in image_types:
# print("A")
#         # Iterate through each image file in each image_type folder
#         # glob reads in any image with the extension "image_dir/im_type/*"
#         # for file in glob.glob(os.path.join(image_dir_training, im_type, "*")):
# for file in glob.glob(os.path.join(image_dir_training, "*")):
#           # print("a")
#           # Call image
#           # img_name = 'monkey.PNG'
#           # img = cv.imread('i:/images/'+img_name)
#     img_real = cv.imread(file)
#     file2 = file.rsplit('/',1)[1]
#     print(file2)
#     img = cv.resize(img_real, (224, 224))
#     WIDTH = img.shape[1]
#     HEIGHT = img.shape[0]
#     x0, x1, y0, y1 = 0, WIDTH - 1, 0, HEIGHT - 1

#     #Image fuzzification
#     layer_b, layer_g, layer_r = cv.split(img)
#     e_layer_b, e_layer_g, e_layer_r = mapping(layer_b, (0, 255), (-1, 1)), mapping(layer_g, (0, 255), (-1, 1)), mapping(layer_r, (0, 255), (-1, 1))
#     e_layer_rgb = scalar_multiplication(1/3, addition(addition(e_layer_b, e_layer_g), e_layer_r)) #Mean of the three layers

#     #Cal Ps, Ws
#     ps, ws = cal_ps_ws(m, n, WIDTH, HEIGHT, GAMMA)

#     #Cal means, variances, lamdas, taos
#     means, variances, lamdas, taos = cal_means_variances_lamdas(ws, e_layer_rgb)

#     #Layers enhacement
#     res_r = one_layer_enhacement(e_layer_r)
#     res_g = one_layer_enhacement(e_layer_g)
#     res_b = one_layer_enhacement(e_layer_b)
#     res_img = cv.merge([res_b, res_g, res_r])
#     # cv2_imshow(res_img)
#     #save image
#     cv.imwrite("/content/drive/My Drive/VR/AVR/COVID_project/FuzzyFied_Images/test/"+file2,res_img)

# """## RGB Color Space Image Enhacement:"""

# #Constants
# n = 2 # number of rows (windows)
# m = 2 # number of colomns (windows)
# GAMMA = 1
# EPSILON = 0.00001
# IDEAL_VARIANCE = 0.35

# #Call image
# img_name = 'monkey.PNG'
# # img = cv.imread('i:/images/'+img_name)
# img_real = cv.imread('/content/drive/My Drive/VR/AVR/COVID_project/Dataset_2/train/Covid-19/1-s2.0-S0929664620300449-gr2_lrg-b.jpg')
# img = cv.resize(img_real, (224, 224))
# WIDTH = img.shape[1]
# HEIGHT = img.shape[0]
# x0, x1, y0, y1 = 0, WIDTH - 1, 0, HEIGHT - 1

# #Image fuzzification
# layer_b, layer_g, layer_r = cv.split(img)
# e_layer_b, e_layer_g, e_layer_r = mapping(layer_b, (0, 255), (-1, 1)), mapping(layer_g, (0, 255), (-1, 1)), mapping(layer_r, (0, 255), (-1, 1))
# e_layer_rgb = scalar_multiplication(1/3, addition(addition(e_layer_b, e_layer_g), e_layer_r)) #Mean of the three layers

# #Cal Ps, Ws
# ps, ws = cal_ps_ws(m, n, WIDTH, HEIGHT, GAMMA)

# #Cal means, variances, lamdas, taos
# means, variances, lamdas, taos = cal_means_variances_lamdas(ws, e_layer_rgb)

# #Layers enhacement
# res_r = one_layer_enhacement(e_layer_r)
# res_g = one_layer_enhacement(e_layer_g)
# res_b = one_layer_enhacement(e_layer_b)
# res_img = cv.merge([res_b, res_g, res_r])

# """### Show the result:"""

# plt.subplot(1, 2, 1)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.subplot(1, 2, 2)
# plt.imshow(cv.cvtColor(res_img, cv.COLOR_BGR2RGB))
# plt.title('Fuzzy RGB image enhacement.')
# plt.show()

def get_fuzzy_images(images, batch_size, n=2, m=3, gamma=1, epsilon=0.00001, ideal_variance=0.35):
    global GAMMA, EPSILON, IDEAL_VARIANCE
    global HEIGHT, WIDTH, x0, x1, y0, y1
    GAMMA = gamma
    EPSILON = epsilon
    IDEAL_VARIANCE = ideal_variance
    res_images = np.empty((images.shape[0], images.shape[1], 0), order='F')
    # images = np.reshape(images, (images.shape[0], images.shape[1], 3, batch_size), order='F')
    print('Fuzzification started....')
    for i in range(batch_size):
        img = images[:, :, :, i]
        # layer = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        WIDTH = img.shape[1]
        HEIGHT = img.shape[0]
        x0, x1, y0, y1 = 0, WIDTH - 1, 0, HEIGHT - 1
        layer_b, layer_g, layer_r = cv.split(img)
        e_layer_b, e_layer_g, e_layer_r = mapping(layer_b, (0, 255), (-1, 1)), mapping(layer_g, (0, 255), (-1, 1)), mapping(layer_r, (0, 255), (-1, 1))
        e_layer_rgb = scalar_multiplication(1/3, addition(addition(e_layer_b, e_layer_g), e_layer_r)) #Mean of the three layers
        print('Ps and Ws calculation is in progress...')
        start = time.time()
        ps, ws = cal_ps_ws(m, n, WIDTH, HEIGHT, GAMMA)
        end = time.time()
        print('Ps and Ws calculation has completed successfully in '+str(end-start)+' s')
        print('means, variances, lamdas and taos calculation is in progress...')
        start = time.time()
        means, variances, lamdas, taos = cal_means_variances_lamdas(ws, e_layer_rgb)
        end = time.time()
        print('means, variances, lamdas and taos calculation is finished in ' + str(end-start) + ' s')
        res_r = one_layer_enhacement(ws, e_layer_r, lamdas, taos)
        res_g = one_layer_enhacement(ws, e_layer_g, lamdas, taos)
        res_b = one_layer_enhacement(ws, e_layer_b, lamdas, taos)
        res_img = cv.merge([res_b, res_g, res_r])
        res_images = np.concatenate((res_images, res_img), axis=2)
    return res_images
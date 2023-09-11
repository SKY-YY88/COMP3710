from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
import glob
#import SimpleITK as sitk
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F

import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
from Data_Loader import Images_Dataset, Images_Dataset_folder
import torchsummary
#from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter

import shutil
import random
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score
import time
#from ploting import VisdomLinePlotter
#from visdom import Visdom


#######################################################
#Checking if GPU is used
#######################################################

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')



os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
device = torch.device("cuda:0" if train_on_gpu else "cpu")

#######################################################
#Setting the basic paramters of the model
#######################################################

batch_size = 8
print('batch_size = ' + str(batch_size))

epoch = 30
print('epoch = ' + str(epoch))

random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))

shuffle = True
valid_loss_min = np.Inf
num_workers = 4
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
epoch_valid = epoch-2
n_iter = 1
i_valid = 0

pin_memory = False
if train_on_gpu:
    pin_memory = True

#plotter = VisdomLinePlotter(env_name='Tutorial Plots')

#######################################################
#Setting up the model
#######################################################

model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]


def model_unet(model_input, in_channel=1, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

#passsing this string so that if it's AttU_Net or R2ATTU_Net it doesn't throw an error at torchSummary


model_test = model_unet(model_Inputs[-1], 1, 1)

model_test.to(device)

#######################################################
#Getting the Summary of Model
#######################################################

torchsummary.summary(model_test, input_size=(1, 128, 128))

#######################################################
#Passing the Dataset of Images and Labels
#######################################################

# t_data = '/flush1/bat161/segmentation/New_Trails/venv/DATA/new_3C_I_ori/'
# l_data = '/flush1/bat161/segmentation/New_Trails/venv/DATA/new_3C_L_ori/'
# test_image = '/flush1/bat161/segmentation/New_Trails/venv/DATA/test_new_3C_I_ori/0131_0009.png'
# test_label = '/flush1/bat161/segmentation/New_Trails/venv/DATA/test_new_3C_L_ori/0131_0009.png'
# test_folderP = '/flush1/bat161/segmentation/New_Trails/venv/DATA/test_new_3C_I_ori/*'
# test_folderL = '/flush1/bat161/segmentation/New_Trails/venv/DATA/test_new_3C_L_ori/*'

t_data = './keras_png_slices_data/keras_png_slices_train/'
l_data = './keras_png_slices_data/keras_png_slices_seg_train/'
test_image = './keras_png_slices_data/keras_png_slices_test/'
test_label = './keras_png_slices_data/keras_png_slices_seg_test/'
test_folderP = './keras_png_slices_data/keras_png_slices_test/*'
test_folderL = './keras_png_slices_data/keras_png_slices_seg_test/*'
valid_image = './keras_png_slices_data/keras_png_slices_validate/'
valid_lable = './keras_png_slices_data/keras_png_slices_seg_validate/'



if torch.cuda.is_available():
    torch.cuda.empty_cache()


data_transform_IN = torchvision.transforms.Compose([
             torchvision.transforms.ToTensor(),
        ])

data_transform = torchvision.transforms.Compose([
             torchvision.transforms.Grayscale(),
        ])
#######################################################
#Loading the model
#######################################################

model_test.load_state_dict(torch.load('./model_2/Unet_D_' +
                   str(epoch) + '_' + str(batch_size)+ '/Unet_epoch_' + str(epoch)
                   + '_batchsize_' + str(batch_size) + '.pth'))

model_test.eval()

#######################################################
#opening the test folder and creating a folder for generated images
#######################################################

read_test_folder = glob.glob(test_folderP)
x_sort_test = natsort.natsorted(read_test_folder)  # To sort

#######################################################
#saving the images in the files
#######################################################

img_test_no = 0

for i in range(len(read_test_folder)):
    im = Image.open(x_sort_test[i])#.convert("RGB")

    im1 = im
    im_n = np.array(im1)
    im_n_flat = im_n.reshape(-1, 1)

    for j in range(im_n_flat.shape[0]):
        if im_n_flat[j] != 0:
            im_n_flat[j] = 255

    s = data_transform_IN(im)
    pred = model_test(s.unsqueeze(0).cuda()).cpu()
    pred = F.sigmoid(pred)
    pred = pred.detach().numpy()

#    pred = threshold_predictions_p(pred) #Value kept 0.01 as max is 1 and noise is very small.

    if i % 24 == 0:
        img_test_no = img_test_no + 1

    x1 = plt.imsave('./model_2/gen_images/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', pred[0][0])


####################################################
#Calculating the Dice Score
####################################################





read_test_folderP = glob.glob('./model_2/gen_images/*')
x_sort_testP = natsort.natsorted(read_test_folderP)


read_test_folderL = glob.glob(test_folderL)
x_sort_testL = natsort.natsorted(read_test_folderL)  # To sort


dice_score123 = 0.0
x_count = 0
x_dice = 0

for i in range(len(read_test_folderP)):

    x = Image.open(x_sort_testP[i])
    s = data_transform(x)
    s = np.array(s)
    s = threshold_predictions_v(s)
    # print(s)
    # print("------------------")

    #save the images
    x1 = plt.imsave('./model_2/pred_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', s)

    y = Image.open(x_sort_testL[i])
    s2 = data_transform(y)
    s3 = np.array(s2)
   # s2 =threshold_predictions_v(s2)
    # print(s3)

    #save the Images
    y1 = plt.imsave('./model_2/label_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', s3)

    total = dice_coeff(s, s3)
    print(total)

    if total <= 0.9:
        # print(x_sort_testP[i])
        # shutil.os.remove(x_sort_testP[i]) delete <0.9 gen_images
        x_count += 1
    if total > 0.9:
        x_dice = x_dice + total
    dice_score123 = dice_score123 + total


# print('Dice Score : ' + str(dice_score123/len(read_test_folderP)))
print(x_count)
print(x_dice)
print('Dice Score : ' + str(float(x_dice/(len(read_test_folderP)-x_count))))


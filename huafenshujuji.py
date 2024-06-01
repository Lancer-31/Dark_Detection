# -*- coding: utf-8 -*-
import os
import random
import shutil
from shutil import copy2

datadir_normal = "/home1/tjh/tjh/Dark_detection/Dark2/3"

all_data = os.listdir(datadir_normal)  # （图片文件夹�
num_all_data = len(all_data)
print("num_all_data: " + str(num_all_data))
index_list = list(range(num_all_data))
# print(index_list)
random.shuffle(index_list)
num = 0

trainDir = "/home1/tjh/tjh/Dark_detection/Dark2/train/3"  # （将训练集放在这个文件夹下）
if not os.path.exists(trainDir):
    os.mkdir(trainDir)
# trainDir1 = "/home/tjh/tjh/data/HR/jiaocha/2/4/"  # （将训练集放在这个文件夹下）
# if not os.path.exists(trainDir1):
#     os.mkdir(trainDir1)
# trainDir2 = "/home/tjh/tjh/data/HR/jiaocha/3/4/"  # （将训练集放在这个文件夹下）
# if not os.path.exists(trainDir2):
#     os.mkdir(trainDir2)
# trainDir3 = "/home/tjh/tjh/data/HR/jiaocha/4/4/"  # （将训练集放在这个文件夹下）
# if not os.path.exists(trainDir3):
#     os.mkdir(trainDir3)
# trainDir4 = "/home/tjh/tjh/data/HR/jiaocha/5/4/"  # （将训练集放在这个文件夹下）
# if not os.path.exists(trainDir4):
#     os.mkdir(trainDir4)
# validDir = '/home/tjh/tjh/data/Fog2/val/4/'  # （将验证集放在这个文件夹下）
# if not os.path.exists(validDir):
#    os.mkdir(validDir)

testDir = '/home1/tjh/tjh/Dark_detection/Dark2/test/3'  # （将测试集放在这个文件夹下）
if not os.path.exists(testDir):
    os.mkdir(testDir)

for i in index_list:
    fileName = os.path.join(datadir_normal, all_data[i])
    if num < num_all_data * 0.7:
        # print(str(fileName))
        copy2(fileName, trainDir)
    # elif num > num_all_data * 0.2 and num < num_all_data * 0.4:
    #     # print(str(fileName))
    #    copy2(fileName, trainDir1)
    # elif num > num_all_data * 0.4 and num < num_all_data * 0.6:
    #     # print(str(fileName))
    #    copy2(fileName, trainDir2)
    # elif num > num_all_data * 0.6 and num < num_all_data * 0.8:
    #     # print(str(fileName))
    #    copy2(fileName, trainDir3)
    # else:
    #     # print(str(fileName))
    #    copy2(fileName, trainDir4)
    # elif num > num_all_data * 0.7 and num < num_all_data * 0.9:
    #     # print(str(fileName))
    #    copy2(fileName, validDir)
    else:
        copy2(fileName, testDir)
    num += 1

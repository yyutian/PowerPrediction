# HDF5文件存放两类对象: dataset (类似数组的数据集)、group (类似词典)
import h5py as h5
import matplotlib.pyplot as plt
import os
import numpy as np

mapHeight = 24
mapWidth = 24
filePath = "./datasets/JN_Fill_2017-2020_M{}x{}_Power.h5".format(mapHeight,mapWidth)
filePath2 = "./datasets/JN_Fill_2017-2020_M{}x{}_Power_Decomposition.h5".format(mapHeight,mapWidth)

file = h5.File(filePath, "r")
file2 = h5.File(filePath2, "r")
#主键值为data 对应的数据赋给data2
data2 = file2['data']
print(file.keys())
data = file['data']
date = file['date']
#print(data.shape, date.shape)
print(data[0][0])
print(data2[0][0])
print(np.sum(data[0][0]==0))

def is_empty(array):
    flag = False
    for i in array:
        for j in i:
            if j != 0:
                flag = True
                break
        if flag:
            break
    # print(flag)
    return flag
pic_path='picture'
if os.path.isdir(pic_path) is False:
    os.mkdir(pic_path)

def plot_image(image, ke):
    fig=plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image)
    #plt.savefig('./picture/Dec_' + str(ke) + '.jpg')
    plt.show()

cont=0
for i in range(len(data2)):
    if(i>100):
        break
    plot_image(data2[i][0], i)
    print(date[i])
    if is_empty(data2[i][0]):
        cont += 1
print(cont)


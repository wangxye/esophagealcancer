# _*_ coding:utf-8 _*_
"""
@Time     : 2022/7/4 10:50
@Author   : Wangxuanye
@File     : ct_dataset_preprocess.py
@Project  : esophagealcancer
@Software : PyCharm
@License  : (C)Copyright 2018-2028, Taogroup-NLPR-CASIA
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/7/4 10:50        1.0             None
"""

import numpy as np
import collections
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import os, sys
import pandas as pd
from keras.preprocessing import image

## Set GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Load model
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import Model

from radiomics import featureextractor

# fc
base_model = ResNet50(weights='imagenet', include_top=True)
from keras.models import Model

model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# %%
# model.summary()
# %%
# Load batch file

imgDir = './data/CECT'
dirlist = os.listdir(imgDir)[:]


# %%
# read images in Nifti format
def loadSegArraywithID(fold, iden):
    path = fold
    pathList = os.listdir(path)
    segPath = [os.path.join(path, i) for i in pathList if ('seg' in i.lower()) & (iden in i.lower())][0]
    seg = sitk.ReadImage(segPath)
    return seg


# read regions of interest (ROI) in Nifti format
def loadImgArraywithID(fold, iden):
    path = fold
    pathList = os.listdir(path)

    imgPath = [os.path.join(path, i) for i in pathList if ('im' in i.lower()) & (iden in i.lower())][0]
    img = sitk.ReadImage(imgPath)
    return img


# Feature Extraction
# Cropping box
def maskcroppingbox(images_array, use2D=False):
    images_array_2 = np.argwhere(images_array)
    print(images_array_2.shape)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = images_array_2.min(axis=0), images_array_2.max(axis=0) + 1
    print(zstart, ystart, xstart)
    print(zstop, ystop, xstop)
    return (zstart, ystart, xstart), (zstop, ystop, xstop)


def featureextraction(imageFilepath, maskFilepath):
    image_array = sitk.ReadImage(imageFilepath)
    image_array = sitk.GetArrayFromImage(image_array)

    mask_array = sitk.ReadImage(maskFilepath)
    mask_array = sitk.GetArrayFromImage(mask_array)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = maskcroppingbox(mask_array, use2D=False)
    roi_images = image_array[zstart - 1:zstop + 1, ystart:ystop, xstart:xstop].transpose((2, 1, 0))
    roi_images1 = zoom(roi_images, zoom=[224 / roi_images.shape[0], 224 / roi_images.shape[1], 1], order=3)
    roi_images2 = np.array(roi_images1, dtype=np.float)
    x = image.img_to_array(roi_images2)
    num = []
    for i in range(zstart, zstop):
        mask_array = np.array(mask_array, dtype='uint8')
        images_array_3 = mask_array[:, :, i]
        num1 = images_array_3.sum()
        num.append(num1)
    maxindex = num.index(max(num))
    # print(max(num), ([*range(zstart, zstop)][num.index(max(num))]))
    x1 = np.asarray(x[:, :, maxindex - 1])
    x2 = np.asarray(x[:, :, maxindex])  # ??????slice
    x3 = np.asarray(x[:, :, maxindex + 1])
    # print(x1.shape)
    x1 = np.expand_dims(x1, axis=0)
    x2 = np.expand_dims(x2, axis=0)
    x3 = np.expand_dims(x3, axis=0)
    a1 = np.asarray(x1)
    a2 = np.asarray(x2)
    a3 = np.asarray(x3)
    # print(a1.shape)
    mylist = [a1, a2, a3]  # ????
    x = np.asarray(mylist)
    # print(x.shape)
    x = np.transpose(x, (1, 2, 3, 0))
    # print(x.shape)
    x = preprocess_input(x)

    base_model_pool_features = model.predict(x)

    features = base_model_pool_features[0]

    deeplearningfeatures = collections.OrderedDict()
    for ind_, f_ in enumerate(features):
        deeplearningfeatures[str(ind_)] = f_
    return deeplearningfeatures


def radiomics_feature_resnet50():
    # %%
    """数据成对的处理，mask and raw"""
    # 读取所有的nii文件，保存文件名字，然后读入gz文件
    res = []
    for i in dirlist:
        if i[-1] == 'i':
            res.append(i)

    # %%
    list_pic = []
    for i in res:
        if i[8] == "_" or i[8] == "-":
            i = i[0:8]
        else:
            i = i[0:9]
        list_pic.append(i)

    # %%
    featureDict = {}
    for i, ind in enumerate(res):

        if i == 2:
            break

        print('处理第{}张图像'.format(i))
        path1 = './data/CECT/' + ind
        # step1: 找到你想要插入位置的索引
        str_index = path1.find('.nii')  # rfind 找出右边索引位置
        # step2: 拼接插入
        path2 = path1[:str_index] + '_1' + path1[str_index:] + '.gz'

        deeplearningfeatures = featureextraction(path1, path2)

        result = deeplearningfeatures
        # print(result)
        key = list(result.keys())
        # print(key)
        key = key[0:]

        feature = []
        for jind in range(len(key)):
            feature.append(result[key[jind]])

        featureDict[list_pic[i]] = feature
        dictkey = key

    # %%
    # dataframe = pd.DataFrame.from_dict(featureDict, orient='index', columns='dictkey')
    dataframe = pd.DataFrame.from_dict(featureDict, orient='index')
    dataframe.to_csv('./data/tmp/resnet50_feature_tmp.csv')

    return dataframe


def radiomics_feature_radiomics():
    # %%
    """数据成对的处理，mask and raw"""
    # 读取所有的nii文件，保存文件名字，然后读入gz文件

    res = []
    for i in dirlist:
        if i[-1] == 'i':
            res.append(i)

    list_pic = []
    for i in res:
        if i[8] == "_" or i[8] == "-":
            i = i[0:8]
        else:
            i = i[0:9]
        list_pic.append(i)

    # %%
    featureDict = {}
    for i, (ind, indd) in enumerate(zip(res, list_pic)):

        # if i == 2:
        #     break

        print('处理第{}张图像'.format(i))
        path1 = 'data/CECT/' + ind
        # step1: 找到你想要插入位置的索引
        str_index = path1.find('.nii')  # rfind 找出右边索引位置
        # step2: 拼接插入
        path2 = path1[:str_index] + '_1' + path1[str_index:] + '.gz'

        # if "P0742731_yangzhengfang1" in path2:
        #     print("delete..")
        #     continue

        extractor = featureextractor.RadiomicsFeatureExtractor()
        print(path1 + "====>" + path2)
        result = extractor.execute(path1, path2)

        key = list(result.keys())
        key = key[22:]
        feature = []
        for jind in range(len(key)):
            feature.append(result[key[jind]])

        featureDict[indd] = feature
        dictkey = key

    dataframe = pd.DataFrame.from_dict(featureDict, orient='index', columns=dictkey)
    dataframe.to_csv('data/tmp/Features_Radiomics_tmp.csv')
    return dataframe


def data_parse():
    new_data = pd.read_excel('data/data_new.xlsx')
    # new_data = pd.read_excel('data/data_new.xlsx', sheet_name="OS")
    radiomics = pd.read_csv('data/tmp/resnet50_feature.csv')
    # radiomics = pd.read_csv('data/tmp/Features_Radiomics.csv')
    # 数据对齐
    radiomics.index = radiomics.iloc[:, 0]
    radiomics.drop('Unnamed: 0', inplace=True, axis=1)

    new_data.index = new_data.iloc[:, 0]
    # 'OS(month)','follow-up time' 'death（yes=1，no=0）'
    new_data.drop(['PatientID', 'patientsname', 'label', 'OS(month)'], inplace=True,
                  axis=1)

    print(str(new_data.shape) + "==>" + str(radiomics.shape))
    data = pd.concat([new_data, radiomics], axis=1)
    data.dropna(inplace=True)

    data.to_csv('data/esophagealcancer_multimodal_data.csv')
    print(data)


def data_parse_OS():
    # new_data = pd.read_excel()
    new_data = pd.read_excel('data/data_new.xlsx', sheet_name="OS")
    radiomics = pd.read_csv('data/tmp/resnet50_feature.csv')
    # radiomics = pd.read_csv('data/tmp/Features_Radiomics.csv')

    print(len(set(new_data['PatientID'])))

    print("=====")
    for index, row in new_data.iterrows():
        if row['PatientID'] not in np.array(radiomics['Unnamed: 0']):
            print(row['PatientID'])
            new_data.drop(index, inplace=True)

    print(len(set(new_data['PatientID'])))

    for i, ri in radiomics.iterrows():
        if ri['Unnamed: 0'] not in np.array(new_data['PatientID']):
            print(ri['Unnamed: 0'])
            radiomics.drop(i, inplace=True)
        # flag = False
        # for j, rj in new_data.iterrows():
        #     if ri['Unnamed: 0'] == rj['PatientID']:
        #         flag = True
        #
        # if flag == False:
        #     print(ri['Unnamed: 0'])
        #     radiomics.drop(i, inplace=True)

    for i in np.where(new_data.duplicated(subset=['PatientID']) == True):
        print(new_data['PatientID'][i])

    new_data.drop_duplicates(subset=['PatientID'], inplace=True)

    print(len(set(new_data['PatientID'])))

    # 数据对齐
    radiomics.index = radiomics.iloc[:, 0]
    radiomics.drop('Unnamed: 0', inplace=True, axis=1)

    new_data.index = new_data.iloc[:, 0]
    # 'OS(month)','follow-up time' 'death（yes=1，no=0）'
    # new_data.drop(['PatientID', 'patientsname', 'OS(month)', 'label'], inplace=True,
    #               axis=1)
    delete_cols = ['PatientID', 'patientsname', 'OS(month)', 'label']
    for s in delete_cols:
        new_data.drop(s, inplace=True,
                      axis=1)

    print(str(new_data.shape) + "==>" + str(radiomics.shape))
    data = pd.concat([new_data, radiomics], axis=1)
    data.dropna(inplace=True)

    # data.to_csv('./data/esophagealcancer_multimodal_data.csv')
    data.to_excel('./data/esophagealcancer_multimodal_data.xlsx', sheet_name="OS", index=False)
    print(data)


if __name__ == '__main__':
    #
    # radiomics_feature_radiomics()
    data_parse_OS()

# Deeplearning-based radiomics
# %%
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
    image_array = sitk.ReadImage(path1)
    image_array = sitk.GetArrayFromImage(image_array)

    mask_array = sitk.ReadImage(path2)
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

'''
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
    path1 = '/data0/BigPlatform/FL/lirongchang/medical/esophagus cancer1/dataset/esophagealcancer_new/CECT/' + ind
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
dataframe.to_csv('./data/tmp/resnet50_feature.csv')
'''
# %%
# 数据分析
import pandas as pd

new_data = pd.read_excel('./data/data_new.xlsx')
radiomics = pd.read_csv('./data/tmp/resnet50_feature.csv')

# %%
# 数据对齐
radiomics.index = radiomics.iloc[:, 0]
# %%
radiomics.drop('Unnamed: 0', inplace=True, axis=1)
# %%
new_data.index = new_data.iloc[:, 0]
# %%
new_data.drop(['PatientID', 'patientsname', 'death（yes=1，no=0）', 'OS(month)', 'follow-up time'], inplace=True, axis=1)

# %%
print(str(new_data.shape) + "==>" + str(radiomics.shape))
data = pd.concat([new_data, radiomics], axis=1)
data.dropna(inplace=True)

# %%
print(data)

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

"""数据无预处理"""
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

aa = ['label']
X = data.drop(aa, axis=1)

# X = new_data.iloc[:,1:]
Y = data.iloc[:, 0].astype('int')

# %%
# 特征重要性分析
# import operator
# from sklearn.ensemble import RandomForestRegressor
#
# rf = RandomForestRegressor()
# rf.fit(X, Y)
# print(sorted(zip(X.columns, map(lambda x: round(x, 4),
#                                                    rf.feature_importances_)),
#              key=operator.itemgetter(1), reverse=True))

# %%
# 选择特征进行预测
# columns1 = ['original_glcm_Correlation','smokingindex',\
#             'original_shape_MinorAxisLength','original_glcm_Imc1','original_glcm_Idmn',\
#             'original_shape_Elongation','original_gldm_DependenceEntropy','original_shape_Maximum2DDiameterSlice',
#             'location(1=upper,2=midian,3=lower)','original_glszm_SizeZoneNonUniformityNormalized',\
#             'original_firstorder_Mean','age','left-recurrent-laryngeal-nerve-involvement(0=no,1=yes)','original_firstorder_RootMeanSquared','surgery（open  Ivor Lewis=1，open McKeown=2，minimally invasive  Ivor Lewis =3，minimally invasive McKeown=4）','original_glcm_Idmn']
# index1 = X.index
#
# #%%
# kong=pd.DataFrame(columns=columns1,index=index1)
# for i in columns1:
#     kong[i] = X[i]
# X = kong

# %%
scv = StratifiedKFold(n_splits=5, random_state=12345, shuffle=True)
models = [
    # LogisticRegression(),\
    LinearDiscriminantAnalysis(), \
    SVC(), \
    GaussianNB(), \
    KNeighborsClassifier(), \
    AdaBoostClassifier(), \
    ]
for dt in models:
    score = cross_val_score(dt, X, Y, scoring='roc_auc', cv=scv)
    print('*' * 100)
    print(dt)
    print("Cross Validation Scores are {}".format(score))
    print("Average Cross Validation score :{}".format(score.mean()))

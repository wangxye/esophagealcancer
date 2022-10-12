# %%
import numpy as np
import collections
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import os, sys
import pandas as pd
from radiomics import featureextractor


def maskcroppingbox(images_array, use2D=False):
    # print(images_array)
    images_array_2 = np.argwhere(images_array)  # 返回值不为0的索引
    # print(images_array_2.min(axis=0))
    (zstart, ystart, xstart), (zstop, ystop, xstop) = images_array_2.min(axis=0), images_array_2.max(axis=0) + 1
    return (zstart, ystart, xstart), (zstop, ystop, xstop)


# %%
"""数据读入"""
imgDir = 'data/CECT'
dirlist = os.listdir(imgDir)[1:]
# print(len(dirlist))
'''
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
dataframe.to_csv('data/tmp/Features_Radiomics.csv')
'''

# %%
# 数据分析
# new_data = pd.read_excel('data/newdata.xlsx')
new_data = pd.read_excel('data/data_new.xlsx')
radiomics = pd.read_csv('data/tmp/Features_Radiomics.csv')

# for i, ri in radiomics.iterrows():
#     flag = False
#     for j, rj in new_data.iterrows():
#         if ri['Unnamed: 0'] == rj['PatientID']:
#             flag = True
#
#     if flag == False:
#         print(ri['Unnamed: 0'])
#         radiomics.drop(i, inplace=True)

# print("=====")
# for index, row in new_data.iterrows():
#     if row['Unnamed: 0'] not in np.array(new_data['PatientID']):
#         print(row['Unnamed: 0'])
#         radiomics.drop(index, inplace=True)


# %%
# 数据对齐
radiomics.index = radiomics.iloc[:, 0]
radiomics.drop('Unnamed: 0', inplace=True, axis=1)

# %%
new_data.index = new_data.iloc[:, 0]
new_data.drop(['PatientID', 'patientsname', 'death（yes=1，no=0）', 'OS(month)', 'follow-up time'], inplace=True, axis=1)

# %%
print(str(new_data.shape) + "==>" + str(radiomics.shape))
data = pd.concat([new_data, radiomics], axis=1)
data.dropna(inplace=True)

# %%
data

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
import operator
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X, Y)
print(sorted(zip(X.columns, map(lambda x: round(x, 4),
                                rf.feature_importances_)),
             key=operator.itemgetter(1), reverse=True))

# %%
# 重要性特征进行筛选
# import xgboost
# from xgboost import plot_importance
# model = xgboost.XGBRegressor(max_depth=30, learning_rate=0.005, n_estimators=10)
# model.fit(X, Y.astype('int'))
#
# fig,ax = plt.subplots(figsize=(25,10))
# plot_importance(model, ax=ax)
# plt.show()

# %%
scv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
models = [LogisticRegression(class_weight='balanced'), \
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

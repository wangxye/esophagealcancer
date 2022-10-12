#%%
import os
import pandas as pd
import numpy as np
'''
读取文件夹下所有文件的名字并把他们用列表存起来
'''
path = "/data0/BigPlatform/FL/lirongchang/medical/esophagus cancer1/dataset/esophagealcancer/CECT"
datanames = os.listdir(path)
list_pic = dict()
for i in datanames:
    if i[8] == "_" or i[8] == "-":
        i = i[0:8]
    else:
        i = i[0:9]

    list_pic[i] = 5

list_pic1 = dict()
for i in datanames:
    if i[8] == "_" or i[8] == "-":
        ii = i[0:8]
    else:
        ii = i[0:9]

    list_pic1[ii] = i

#%%
data = pd.read_excel('/data0/BigPlatform/FL/lirongchang/medical/esophagus cancer1/esophagus cancer1/data_new.xlsx')
list_id = data.iloc[0:,0]
list_label = data.iloc[0:,5]
list_idlabel = [np.array(list_id), np.array(list_label)]#np.concatenate([list_id, list_label], axis=1)


#%%
j = 0
for i in range(len(list_idlabel[0])):
    if str(list_idlabel[0][i]) in list_pic:

        old = '/data0/BigPlatform/FL/lirongchang/medical/esophagus cancer1/dataset/esophagealcancer/CECT/' + str(list_pic1[list_idlabel[0][i]])
        new = '/data0/BigPlatform/FL/lirongchang/medical/esophagus cancer1/dataset/esophagealcancer/CECT/' + str(list_idlabel[0][i]) + str(list_idlabel[1][i]) + '.nii.gz'

        print(old)
        # print(new)
        try:
            os.rename(old, new)
        except:
            # print(old)
            print('1')
        j+=1
    # else:
    #     print('list_idlabel[0][i]')
print(j)
#%%
# old = r'D:\1\P0626081_getianping.nii'
# new = 'D:\\1\\'+'111.nii'
# print(old)
# os.rename(old, new)
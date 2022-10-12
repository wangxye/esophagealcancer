# %%
import numpy as np
import sklearn
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

data1 = pd.read_excel('data/newdata.xlsx',
                      header=0)
data1.head()
target = np.array(data1)[:, 5].astype('int')
feature = np.array(data1)[:, 6:]

seed = 12345
np.random.seed(seed)

# %%
"""数据分析步骤"""
# 重要性特征进行筛选
import xgboost
from xgboost import plot_importance

model = xgboost.XGBRegressor(max_depth=30, learning_rate=0.005, n_estimators=1000)
model.fit(data1.iloc[:, 6:], data1.iloc[:, 5].astype('int'))
fig, ax = plt.subplots(figsize=(15, 8))
plot_importance(model, ax=ax)
plt.show()

# %%
"""数据特征的处理方法"""
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt

cols_standardize = ['age', 'smokingindex']
cols_binary = []
# cols_categorical = ['family-history-of-cancer(0=no,1=yes)', 'esophagectomy(0=open,1=minimally invasive)', \
#                     'approach（ Ivor Lewis technique=0，McKeown technique=1）', 'Tstages', \
#                     'differentiation(1=low,2=medium,3=high)']
cols_categorical = ['esophagectomy(0=open,1=minimally invasive)', \
                    'Tstages', \
                    'differentiation(1=low,2=medium,3=high)']

'''
Feature transform
Standardize numerical variables, keep binary variables unchanged, and perform Entity Embedding on categorical variables
'''

standardize = [([col], StandardScaler()) for col in cols_standardize]
binary = [(col, None) for col in cols_binary]
categorical = [(col, OrderedCategoricalLong()) for col in cols_categorical]
x_mapper_float = DataFrameMapper(standardize + binary)
x_mapper_long = DataFrameMapper(categorical)

x_fit_transform = lambda df: tt.tuplefy(x_mapper_float.fit_transform(df), x_mapper_long.fit_transform(df))
x_transform = lambda df: tt.tuplefy(x_mapper_float.transform(df), x_mapper_long.transform(df))

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

# %%
"""数据无预处理"""

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

print(25 * '*' + "无预处理" + 25 * '*')
Y = np.array(data1)[:, 5].astype('int')
X = np.array(data1)[:, 6:]
scv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
# dt = LogisticRegression(class_weight='balanced')
# dt = KNeighborsClassifier()
models = [LogisticRegression(class_weight='balanced'), \
          LinearDiscriminantAnalysis(), \
          SVC(), \
          GaussianNB(), \
          KNeighborsClassifier(), \
          AdaBoostClassifier(), \
          ]
for dt in models:
    score = cross_val_score(dt, X, Y, scoring='roc_auc', cv=scv, n_jobs=-1)
    print('*' * 100)
    print(dt)
    print("Cross Validation Scores are {}".format(score))
    print("Average Cross Validation score :{}".format(score.mean()))

# %%
"""数据预处理"""
print(25 * '*' + "预处理" + 25 * '*')
X = x_fit_transform(data1.drop('label', axis=1))
X = np.append(np.array(X[0].astype('float32')), np.array(X[1]), axis=1)
Y = np.array(data1['label'])
scv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
# dt = LogisticRegression(class_weight='balanced')
# dt = DecisionTreeClassifier()
# dt = GradientBoostingClassifier()

models = [LogisticRegression(class_weight='balanced'), \
          LinearDiscriminantAnalysis(), \
          SVC(), \
          GaussianNB(), \
          KNeighborsClassifier(), \
          AdaBoostClassifier(), \
          ]
for dt in models:
    score = cross_val_score(dt, X, Y, scoring='roc_auc', cv=scv, n_jobs=-1)
    print('*' * 100)
    print(dt)
    print("Cross Validation Scores are {}".format(score))
    print("Average Cross Validation score :{}".format(score.mean()))

# score = cross_val_score(dt, X, Y, scoring='roc_auc', cv=scv, n_jobs=-1)
# print('*' * 100)
# print("Cross Validation Scores are {}".format(score))
# print("Average Cross Validation score :{}".format(score.mean()))

# %%
"""博客实验"""
from imblearn.pipeline import make_pipeline
from imblearn import datasets
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold
from sklearn.metrics import recall_score, roc_auc_score

import pandas as pd
import numpy as np

x = x_fit_transform(data1.iloc[:, 6:])
x = np.append(np.array(x[0].astype('float32')), np.array(x[1]), axis=1)
X_train, X_test, y_train, y_test = train_test_split( \
    x, np.array(data1['label']), test_size=0.1, random_state=seed)

pred_list = []
truth_list = []
res_list = []


def score_model(model, cv=None):
    if cv is None:
        cv = KFold(n_splits=10, random_state=42, shuffle=True)

    smo = SMOTE(random_state=42)  # 过采样

    # from imblearn.under_sampling import ClusterCentroids
    # smo = ClusterCentroids(random_state=42) # 欠采样

    scores = []

    for train_fold_index, val_fold_index in cv.split(X_train, y_train):
        X_train_fold, y_train_fold = X_train[train_fold_index], y_train[train_fold_index]
        X_val_fold, y_val_fold = X_train[val_fold_index], y_train[val_fold_index]

        X_train_fold_upsample, y_train_fold_upsample = smo.fit_resample(X_train_fold,
                                                                        y_train_fold)
        model_obj = model().fit(X_train_fold_upsample, y_train_fold_upsample)
        score = roc_auc_score(y_val_fold, model_obj.predict(X_val_fold))
        pred = model_obj.predict(X_val_fold)
        pred_list.append(pred)
        truth_list.append(y_val_fold)
        res_list.append(score)

        scores.append(score)
    return np.array(scores)


score = score_model(DecisionTreeClassifier)
print('*' * 100)
print("Cross Validation Scores are {}".format(score))
print("Average Cross Validation score :{}".format(score.mean()))

# %%
from sklearn.metrics import confusion_matrix, classification_report

X = x_fit_transform(data1.drop('label', axis=1))
X = np.append(np.array(X[0].astype('float32')), np.array(X[1]), axis=1)
Y = np.array(data1['label'])

X_train, X_test, y_train, y_test = train_test_split( \
    X, Y, test_size=0.1, random_state=False)

scv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
dt = LogisticRegression(class_weight='balanced')
model = dt.fit(X_train, y_train)
predictions = model.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(predictions)
print(roc_auc_score(y_test, predictions))
# print('*'*100)
# print("Cross Validation Scores are {}".format(score))
# print("Average Cross Validation score :{}".format(score.mean()))

# %%
import numpy as np
import matplotlib.pyplot as plt

classes = ['0', '1']
font_dict = dict(fontsize=30,
                 color='r',
                 family='Times New Roman',
                 weight='light',
                 style='italic',
                 )

for y_test, predictions, auc in zip(truth_list, pred_list, res_list):
    from sklearn.metrics import confusion_matrix

    confusion_matrix = confusion_matrix(y_test, predictions)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = confusion_matrix.max() / 2.

    iters = np.reshape([[[i, j] for j in range(2)] for i in range(2)], (confusion_matrix.size, 2))
    for i, j in iters:
        plt.text(j, i, format(confusion_matrix[i, j]), fontdict=font_dict)  # 显示对应的数字

    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.title('the auc is {}'.format(auc))
    plt.tight_layout()
    plt.show()

# %%
for y_test, predictions, auc in zip(truth_list, pred_list, res_list):
    print('auc', auc)
    print('预测标签', predictions)
    print('真实标签', y_test)
    print(' ')

# %%
"""pca降维分析数据"""
from sklearn.metrics import confusion_matrix, classification_report

X = x_fit_transform(data1.drop('label', axis=1))
X = np.append(np.array(X[0].astype('float32')), np.array(X[1]), axis=1)
Y = np.array(data1['label'])

from sklearn.decomposition import PCA

pca = PCA(n_components=2)  # 实例化
pca = pca.fit(X)  # 拟合模型
x_dr = pca.transform(X)  # 获取新矩阵
y = Y
x_dr[y == 0, 0]  # 采用布尔索引

# 画出分类图
plt.figure()  # 创建一个画布
plt.scatter(x_dr[y == 0, 0], x_dr[y == 0, 1], c="red", label='class 0')
plt.scatter(x_dr[y == 1, 0], x_dr[y == 1, 1], c="black", label='class 1')
# plt.scatter(x_dr[y==2,0],x_dr[y==2,1],c="orange",label = iris.target_names[2])
plt.legend()  # 显示图例
plt.title("PCA of medical dataset")  # 显示标题
plt.show()

# %%
from sklearn.manifold import TSNE


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] * 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


tsne = TSNE(n_components=2, init='pca', random_state=0)
result = tsne.fit_transform(X)
fig = plot_embedding(result, Y,
                     't-SNE embedding of the digits'
                     )
# %%
plt.show()

# -*-codeing = utf-8 -*-
# @Time :2022/6/27 14:47
# @Author :Wangxuanye
# @File :ct_model_train_dl.py.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import xgboost
from xgboost import plot_importance

"""数据无预处理"""
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression

from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def prepare_data():
    # %%
    # 数据分析
    # new_data = pd.read_excel('data/newdata.xlsx')
    new_data = pd.read_excel('data/data_new.xlsx')
    radiomics = pd.read_csv('data/tmp/resnet50_feature.csv')

    # %%
    # 数据对齐
    radiomics.index = radiomics.iloc[:, 0]
    radiomics.drop('Unnamed: 0', inplace=True, axis=1)

    # %%
    new_data.index = new_data.iloc[:, 0]
    new_data.drop(['PatientID', 'patientsname', 'death（yes=1，no=0）', 'OS(month)', 'follow-up time'], inplace=True,
                  axis=1)

    # %%
    print(str(new_data.shape) + "==>" + str(radiomics.shape))

    data = pd.concat([new_data, radiomics], axis=1)
    data.dropna(inplace=True)
    print(data)
    return data


def train_no_tranform(data):
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
    scv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    models = [LogisticRegression(), \
              LinearDiscriminantAnalysis(), \
              SVC(), \
              GaussianNB(), \
              KNeighborsClassifier(), \
              AdaBoostClassifier(), \
              RandomForestClassifier(n_estimators=121), \
              GradientBoostingClassifier(n_estimators=1)
              ]
    for dt in models:
        score = cross_val_score(dt, X, Y, scoring='roc_auc', cv=scv)
        print('*' * 100)
        print(dt)
        print("Cross Validation Scores are {}".format(score))
        print("Average Cross Validation score :{}".format(score.mean()))


def feature_transform():
    cols_standardize = ['age', 'smokingindex',
                        # 'original_glcm_Correlation', 'original_shape_MinorAxisLength', \
                        # 'original_glcm_Idmn', 'original_gldm_DependenceEntropy', \
                        # 'original_shape_Flatness', 'original_firstorder_Energy', \
                        #
                        # 'original_shape_MeshVolume',
                        # 'original_shape_SurfaceArea', 'original_shape_VoxelVolume',
                        # 'original_firstorder_Range', 'original_firstorder_Variance',
                        ]
    #
    cols_binary = []
    # cols_categorical = ['family-history-of-cancer(0=no,1=yes)', 'esophagectomy(0=open,1=minimally invasive)', \
    #                     'approach（ Ivor Lewis technique=0，McKeown technique=1）', 'Tstages', \
    #                     'differentiation(1=low,2=medium,3=high)']
    cols_categorical = ['esophagectomy(0=open,1=minimally invasive)', \
                        'approach（ Ivor Lewis technique=0，McKeown technique=1）', \
                        'Tstages', \
                        'differentiation(1=low,2=medium,3=high)', \
                        ]

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

    return x_fit_transform, x_transform


def feature_analysis(data1):
    model = xgboost.XGBRegressor(max_depth=30, learning_rate=0.005, n_estimators=1000)
    model.fit(data1.iloc[:, 1:], data1.iloc[:, 0].astype('int'))
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_importance(model, ax=ax, max_num_features=15)
    # print(model.importance_type)
    print(model.feature_importances_)
    plt.show()

    # corr = data.corr()
    # plt.figure(figsize=(14, 14))
    # # annot=True 显示每个方格的数据
    # sns.heatmap(corr, annot=True)
    # plt.show()
    from sklearn.feature_selection import SelectFromModel
    sfm_selector = SelectFromModel(estimator=LogisticRegression())
    sfm_selector.fit(data1.iloc[:, 1:], data1.iloc[:, 0])
    print(data1.iloc[:, 1:].columns[sfm_selector.get_support()])


def CT_T_checkout(tData):
    from scipy.stats import levene, ttest_ind

    classinformation = tData["label"].unique()

    for temp_classinformation in classinformation:
        temp_data = tData[tData['label'].isin([temp_classinformation])]
        exec("df%s=temp_data" % temp_classinformation)

    df0 = tData.loc[tData["label"] == 0]
    df1 = tData.loc[tData["label"] == 1]

    counts = 0
    columns_index = []
    for column_name in tData.columns[25:]:
        if levene(df1[column_name], df0[column_name])[1] > 0.05:
            if ttest_ind(df1[column_name], df0[column_name], equal_var=True)[1] < 0.05:
                columns_index.append(column_name)
        else:
            if ttest_ind(df1[column_name], df0[column_name], equal_var=False)[1] < 0.05:
                columns_index.append(column_name)

    print("T检验筛选后剩下的特征数：{}个".format(len(columns_index)))
    # print(columns_index)
    # 数据只保留从T检验筛选出的特征数据，重新组合成data

    # if not 'label' in columns_index:
    #     columns_index = ['label'] + columns_index
    # if not 'index' in columns_index:
    #     columns_index = ['index'] + columns_index
    print(columns_index)
    # df1 = df1[columns_index]
    # df0 = df0[columns_index]
    #
    # tData = pd.concat([df1, df0])
    return tData[columns_index]


def T_checkout_LASSO(tData):
    from scipy.stats import levene, ttest_ind
    from sklearn.linear_model import LassoCV

    classinformation = tData["label"].unique()

    for temp_classinformation in classinformation:
        temp_data = tData[tData['label'].isin([temp_classinformation])]
        exec("df%s=temp_data" % temp_classinformation)

    df0 = tData.loc[tData["label"] == 0]
    df1 = tData.loc[tData["label"] == 1]

    counts = 0
    columns_index = []
    # for column_name in tData.columns[25:]:
    for column_name in tData.columns[1:]:
        if levene(df1[column_name], df0[column_name])[1] > 0.05:
            if ttest_ind(df1[column_name], df0[column_name], equal_var=True)[1] < 0.05:
                columns_index.append(column_name)
        else:
            if ttest_ind(df1[column_name], df0[column_name], equal_var=False)[1] < 0.05:
                columns_index.append(column_name)

    print("T检验筛选后剩下的特征数：{}个".format(len(columns_index)))

    print(columns_index)

    X = data[columns_index]

    X = X.apply(pd.to_numeric, errors='ignore')  # 将数据类型转化为数值型
    colNames = X.columns  # 读取特征的名字
    X = X.fillna(0)  # NaN 值填 0
    X = X.astype(np.float64)  # 转换 float64 类型，防止报 warning
    X = StandardScaler().fit_transform(X)  # 数据矩阵归一化处理（把所有特征的分布压缩在一个可比的范围，以得到可比的特征系数）
    X = pd.DataFrame(X)
    X.columns = colNames

    y = data.iloc[:, 0].astype('int')
    print(X.head())

    alphas = np.logspace(-4, 1, 50)
    # alphas 实际上是 λ 值，常量，通过模型优化选择，但可以给定范围，10e-4 到 1 范围内，等分 50 份，以 log 为间隔（以 10 为底，以等差数列中的每个值为指数），取 50 个值。

    model_lassoCV = LassoCV(alphas=alphas, max_iter=100000).fit(X, y)
    # max_iter 最大迭代数

    coef = pd.Series(model_lassoCV.coef_, index=X.columns)
    # 以 LASSO 计算的系数做一个序列，并以 X.columns 特征名为名称

    print(model_lassoCV.alpha_)  # 选出的最优 alpha 值 0.00040949150623804275
    print('%s %d' % ('Lasso picked', sum(coef != 0)))  # 系数不等于 0 的特殊个数 Lasso picked 21
    print(coef[coef != 0])  # 输出系数不等于 0 的名称和相应系数
    index = coef[coef != 0].index
    X = X[index]

    print("LASSO降维：" + index)
    return X


def LASSO_dimension_reduction(data):
    from sklearn.linear_model import LassoCV
    X = data.iloc[:, 25:]

    X = X.apply(pd.to_numeric, errors='ignore')  # 将数据类型转化为数值型
    colNames = X.columns  # 读取特征的名字
    X = X.fillna(0)  # NaN 值填 0
    X = X.astype(np.float64)  # 转换 float64 类型，防止报 warning
    X = StandardScaler().fit_transform(X)  # 数据矩阵归一化处理（把所有特征的分布压缩在一个可比的范围，以得到可比的特征系数）
    X = pd.DataFrame(X)
    X.columns = colNames

    y = data.iloc[:, 0].astype('int')
    print(X.head())

    # LASSO 特征筛选
    alphas = np.logspace(-4, 1, 50)
    # alphas 实际上是 λ 值，常量，通过模型优化选择，但可以给定范围，10e-4 到 1 范围内，等分 50 份，以 log 为间隔（以 10 为底，以等差数列中的每个值为指数），取 50 个值。

    model_lassoCV = LassoCV(alphas=alphas, max_iter=100000).fit(X, y)
    # max_iter 最大迭代数

    coef = pd.Series(model_lassoCV.coef_, index=X.columns)
    # 以 LASSO 计算的系数做一个序列，并以 X.columns 特征名为名称

    print(model_lassoCV.alpha_)  # 选出的最优 alpha 值 0.00040949150623804275
    print('%s %d' % ('Lasso picked', sum(coef != 0)))  # 系数不等于 0 的特殊个数 Lasso picked 21
    print(coef[coef != 0])  # 输出系数不等于 0 的名称和相应系数
    index = coef[coef != 0].index
    X = X[index]

    print("LASSO降维：" + index)
    return X


def train_transform(data, x_fit_transform, x_transform):
    print(25 * '*' + "预处理" + 25 * '*')

    # aa = ['label']
    # X = data.drop(aa, axis=1)
    # # X = new_data.iloc[:,1:]
    # Y = data.iloc[:, 0].astype('int')

    Y = np.array(data['label'])

    X = x_fit_transform(data.drop('label', axis=1))
    X = np.append(np.array(X[0].astype('float32')), np.array(X[1]), axis=1)

    # X_CT = PCA_dimension_reduction(data)
    # X = PCA_dimension_reduction(data)

    # X_CT = CT_T_checkout(data)
    X_CT = T_checkout_LASSO(data)

    # X_CT = LASSO_dimension_reduction(data)

    # X = np.concatenate((X, X_CT), axis=1)
    X = X_CT
    print(X.shape)
    # print(pca.explained_variance_ratio_)

    scv = StratifiedKFold(n_splits=8, random_state=0, shuffle=True)

    # %%
    models = [
        LogisticRegression(), \
        LinearDiscriminantAnalysis(), \
        SVC(kernel='rbf', gamma='scale', probability=True, class_weight={1: 10}, C=5), \
        SVC(kernel='rbf', gamma='scale', C=1), \
        GaussianNB(), \
        KNeighborsClassifier(), \
        AdaBoostClassifier(), \
        RandomForestClassifier(n_estimators=121, criterion='entropy', class_weight='balanced'), \
        RandomForestClassifier(max_depth=7, min_samples_split=10, n_estimators=71, criterion='entropy'), \
        RandomForestClassifier(max_depth=6, min_samples_split=2, n_estimators=141, criterion='gini'), \
        RandomForestClassifier(max_depth=10, min_samples_split=6, n_estimators=61, criterion='gini'), \
        GradientBoostingClassifier(n_estimators=31)
    ]

    for dt in models:
        score = cross_val_score(dt, X, Y, scoring='roc_auc', cv=scv, n_jobs=-1)
        print('*' * 100)
        print(dt)
        print("Cross Validation Scores are {}".format(score))
        print("Average Cross Validation score :{}".format(score.mean()))

    score_list = []
    for i in range(0, 200, 10):
        # rm = RandomForestClassifier(n_estimators=i + 1, max_depth=6, min_samples_split=2, criterion='gini'
        #                             )
        # rm = RandomForestClassifier(n_estimators=i + 1, max_depth=7, min_samples_split=10, criterion='entropy')
        rm = RandomForestClassifier(n_estimators=i + 1, max_depth=10, min_samples_split=6, criterion='gini')
        # 交叉验证
        score = cross_val_score(rm, X, Y, scoring='roc_auc', cv=scv, n_jobs=-1).mean()
        score_list.append(score)  # 记录每个n_estimators下的精确度

    print((max(score_list), (score_list.index(max(score_list)) * 10) + 1))  # 输出最大的n_estimators以及它的下标
    plt.figure(figsize=[20, 5])  # 展示画布的大小
    plt.plot(range(1, 201, 10), score_list, color="r", label="random predict")
    plt.legend()  # 展示图例
    plt.show()

    sl = []
    for i in range(0, 200, 10):
        rm = GradientBoostingClassifier(n_estimators=i + 1
                                        )
        # 交叉验证
        score = cross_val_score(rm, X, Y, scoring='roc_auc', cv=scv, n_jobs=-1).mean()
        sl.append(score)  # 记录每个n_estimators下的精确度

    print((max(sl), (sl.index(max(sl)) * 10) + 1))  # 输出最大的n_estimators以及它的下标
    plt.figure(figsize=[20, 5])  # 展示画布的大小
    plt.plot(range(1, 201, 10), sl, color="r", label="random predict")
    plt.legend()  # 展示图例
    plt.show()

    # parameters = {'kernel': ('rbf', 'linear'), 'C': [1, 5, 10], "gamma": [1, 0.1, 0.01]}
    parameters = {'kernel': ('rbf', 'linear'), 'C': [1, 5, 10], "gamma": [1, 0.1, 0.01]}
    svr = SVC()
    clf = GridSearchCV(svr, parameters, scoring='roc_auc', cv=scv)
    clf.fit(X, Y)
    # print(clf.best_estimator_)
    print("{},{},{}".format(clf.best_estimator_, clf.best_params_, clf.best_score_))


    # param_test2 = {'max_features': range(5, 30, 1), 'min_samples_leaf': range(1, 11, 1)}
    #
    # gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=31,
    #                                                          max_depth=7,
    #                                                          min_samples_split=10,
    #                                                          ),
    #                         param_grid=param_test2, scoring='roc_auc', cv=scv)

    # param_test2 = {'n_estimators': range(1, 201, 10), 'max_depth': range(1, 20, 1),
    #                'min_samples_split': range(50, 201, 20),
    #                # 'max_features': range(5, 30, 1), 'min_samples_leaf': range(1, 11, 1),
    #                'min_samples_split': range(2, 22, 1), 'criterion': ['gini', 'entropy']}
    # gsearch2 = GridSearchCV(estimator=RandomForestClassifier(),
    #                         param_grid=param_test2, scoring='roc_auc', cv=scv)
    #
    # gsearch2.fit(X, Y)
    # # print(gsearch2.best_params_)
    #
    # print("{},{},{}".format(gsearch2.best_estimator_, gsearch2.best_params_, gsearch2.best_score_))


def PCA_dimension_reduction(data):
    pca = PCA(n_components=0.99)  # 实例化
    X = data.iloc[:, 25:]
    # X_CT = data.iloc[:, 1:]

    colNames = X.columns
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X)
    X.columns = colNames
    print(X.head())

    pca = pca.fit(X)  # 拟合模型
    X_CT = pca.transform(X)  # 获取新矩阵

    print(sum(pca.explained_variance_ratio_))
    print("PCA降维：" + len(X_CT))
    return X_CT


if __name__ == '__main__':
    data = prepare_data()

    # feature_analysis(data)
    # train_no_tranform(data)

    x_fit_transform, x_transform = feature_transform()

    # data_split()

    train_transform(data, x_fit_transform, x_transform)

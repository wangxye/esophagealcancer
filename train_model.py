# -*-codeing = utf-8 -*-
# @Time :2022/6/23 9:36
# @Author :Wangxuanye
# @File :train_model.py
# @Software: PyCharm
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
from xgboost import plot_importance
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from imblearn.pipeline import make_pipeline
from imblearn import datasets
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, classification_report, accuracy_score
from pycox.models import LogisticHazard, CoxCC, DeepHit, MTLR

from DeepSurvivalNet import NetAESurv, LossAELogHaz, Concordance

seed = 12345
np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description='hyper parameters.'
    )
    parser.add_argument('--lr', type=float, default=0.00026388849263575596, help="learning rate")  #
    parser.add_argument('--alpha', type=float, default=0.6713701815022571, help="loss weight")  # 0.7769056354193079
    parser.add_argument('--batch_size', type=int, default=16)  # 32
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.2)  # 0.2
    parser.add_argument('--encoded_features', type=int, default=4)  # 16
    parser.add_argument('--duration', type=str, default="OS", help="name of duration")
    parser.add_argument('--event', type=str, default="OSR", help="name of event")
    parser.add_argument('--data_file', type=str, default="data/newdata.xlsx")
    parser.add_argument('--sheet_name', type=str, default="OS")
    parser.add_argument('--hidden_dim', type=int, default=[96, 80, 112, 80])

    return parser.parse_args()


args = parse_args()


def prepare_data():
    data1 = pd.read_excel('data/newdata.xlsx',
                          header=0)
    data1.head()
    target = np.array(data1)[:, 5].astype('int')
    feature = np.array(data1)[:, 6:]

    return data1, target, feature


def feature_analysis(data1):
    model = xgboost.XGBRegressor(max_depth=30, learning_rate=0.005, n_estimators=1000)
    model.fit(data1.iloc[:, 6:], data1.iloc[:, 5].astype('int'))
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_importance(model, ax=ax)
    plt.show()


def feature_transform():
    cols_standardize = ['age', 'smokingindex']
    cols_binary = []
    # cols_categorical = ['family-history-of-cancer(0=no,1=yes)', 'esophagectomy(0=open,1=minimally invasive)', \
    #                     'approach（ Ivor Lewis technique=0，McKeown technique=1）', 'Tstages', \
    #                     'differentiation(1=low,2=medium,3=high)']
    # cols_categorical = ['esophagectomy(0=open,1=minimally invasive)', \
    #                     'Tstages', \
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


def train_no_transform(data1):
    print(25 * '*' + "无预处理" + 25 * '*')
    # Y = np.array(data1)[:, 5].astype('int')
    # X = np.array(data1)[:, 6:]
    scv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

    Y = np.array(data1)[:, 5].astype('int')
    X = np.array(data1)[:, 6:]
    X_train, X_test, y_train, y_test = train_test_split( \
        X, Y, train_size=0.875, test_size=0.125, random_state=188)

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
        print('*' * 100)
        print(dt)

        score = cross_val_score(dt, X, Y, scoring='roc_auc', cv=scv, n_jobs=-1)
        print("Cross Validation Scores are {}".format(score))
        print("Average Cross Validation score :{}".format(score.mean()))

        # model = dt.fit(X_train, y_train)
        # predictions = model.predict(X_test)
        # print('*' * 100)
        # print(dt)
        # score = roc_auc_score(y_test, predictions)
        # print('--- report ---')
        # print(classification_report(y_test, predictions))
        # print("Test score :{}".format(score))


def polynomial_model(degree=1, **kwarg):
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    logistic_regression = LogisticRegression(**kwarg)
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("logistic_regression", logistic_regression)])
    return pipeline


def train_transform(data1, target, feature, x_fit_transform, x_transform):
    print(25 * '*' + "预处理" + 25 * '*')

    # test
    # x = x_fit_transform(data1.iloc[:, 6:])
    # x = np.append(np.array(x[0].astype('float32')), np.array(x[1]), axis=1)
    # X_train, X_test, y_train, y_test = train_test_split( \
    #     x, target, train_size=0.875, test_size=0.125, random_state=188)

    X = x_fit_transform(data1.drop('label', axis=1))
    X = np.append(np.array(X[0].astype('float32')), np.array(X[1]), axis=1)
    Y = np.array(data1['label'])

    scv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

    # dt = LogisticRegression(class_weight='balanced')
    # dt = DecisionTreeClassifier()
    # dt = GradientBoostingClassifier()

    models = [
        LogisticRegression(), \
        LinearDiscriminantAnalysis(), \
        SVC(), \
        GaussianNB(), \
        KNeighborsClassifier(), \
        AdaBoostClassifier(), \
        RandomForestClassifier(n_estimators=121), \
        #131
        GradientBoostingClassifier(n_estimators=1)
        #11
    ]

    for dt in models:
        print('*' * 100)
        print(dt)
        score = cross_val_score(dt, X, Y, scoring='roc_auc', cv=scv, n_jobs=-1)
        print("Cross Validation Scores are {}".format(score))
        print("Average Cross Validation score :{}".format(score.mean()))

        # test
        # model = dt.fit(X_train, y_train)
        # train_score = model.score(X_train, y_train)  # 训练的得分
        # test_score = model.score(X_test, y_test)  # 测试得分
        # print('train score:{train_score:.6f};test score:{test_score:.6f}'.format(train_score=train_score,
        #                                                                          test_score=test_score))
        # predictions = model.predict(X_test)
        # auc = roc_auc_score(y_test, predictions)
        # print('AUC:', auc)
        #
        # y_pred = model.predict(X_test)  # 进行预测
        # print('matchs:{0}/{1}'.format(np.equal(y_pred, y_test).shape[0], y_test.shape[0]))  # test集的预测对比

        # predictions = model.predict(X_test)
        # score = roc_auc_score(y_test, predictions)
        # print('--- report ---')
        # print(classification_report(y_test, predictions))
        # print("Test score :{}".format(score))

    # print(25 * '*' + "优化" + 25 * '*')
    # parameters = {
    #     'gamma': np.linspace(0.0001, 0.1),
    #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    # }
    #
    # svm_model1 = SVC()
    # # 寻找最优参数
    # grid_model = GridSearchCV(svm_model1, parameters, cv=scv, return_train_score=True)
    # score = cross_val_score(grid_model, X, Y, scoring='roc_auc', cv=scv, n_jobs=-1)
    # print("Cross Validation Scores are {}".format(score))
    # print("Average Cross Validation score :{}".format(score.mean()))
    # # 模型最优参数0
    # # print(svm_model1.best_params_)
    #
    # param_grid = {
    #     "max_depth": np.arange(5, 30)
    # }
    # rm_new = RandomForestClassifier(n_estimators=131
    #                                 , n_jobs=-1
    #                                 )
    # GV = GridSearchCV(rm_new, param_grid=param_grid, cv=10)
    # score = cross_val_score(GV, X, Y, scoring='roc_auc', cv=scv, n_jobs=-1)
    # print("Cross Validation Scores are {}".format(score))
    # print("Average Cross Validation score :{}".format(score.mean()))
    # 模型最优参数
    # print(svm_model1.best_params_)

    # print(25 * '*' + "优化" + 25 * '*')
    # parameters = {
    #     'gamma': np.linspace(0.0001, 0.1),
    #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    # }
    #
    # svm_model1 = SVC()
    # # 寻找最优参数
    # grid_model = GridSearchCV(svm_model1, parameters, cv=10, return_train_score=True)
    # grid_model.fit(X_train, y_train)
    # pred_label = grid_model.predict(X_test)
    # # 模型最优参数
    # print(grid_model.best_params_)
    # # 输出准确率
    # print('准确率:', accuracy_score(pred_label, y_test))
    #
    # auc = roc_auc_score(y_test, pred_label)
    # print('AUC:', auc)
    #
    # param_grid = {
    #     "max_depth": np.arange(5, 30)
    # }
    # rm_new = RandomForestClassifier(n_estimators=131
    #                                 , random_state=90
    #                                 , n_jobs=1
    #                                 )
    # GV = GridSearchCV(rm_new, param_grid=param_grid, cv=10)
    # GV.fit(X_train, y_train)
    # pred = GV.predict(X_test)
    # # 模型最优参数
    # print(GV.best_params_)
    # # 输出准确率
    # print('准确率:', accuracy_score(pred, y_test))
    #
    # auc = roc_auc_score(y_test, pred)
    # print('AUC:', auc)

    # 调参
    # score_list = []
    # for i in range(0, 200, 10):
    #     rm = GradientBoostingClassifier(n_estimators=i + 1
    #                                     )
    #     # 交叉验证
    #     # score = cross_val_score(dt, X, Y, scoring='roc_auc', cv=scv, n_jobs=-1)
    #     # score = cross_val_score(rm, X, Y, cv=10).mean()
    #     score = cross_val_score(rm, X, Y, scoring='roc_auc', cv=scv, n_jobs=-1).mean()
    #     score_list.append(score)  # 记录每个n_estimators下的精确度
    #
    # print((max(score_list), (score_list.index(max(score_list)) * 10) + 1))  # 输出最大的n_estimators以及它的下标
    # plt.figure(figsize=[20, 5])  # 展示画布的大小
    # plt.plot(range(1, 201, 10), score_list, color="r", label="random predict")
    # plt.legend()  # 展示图例
    # plt.show()

    score_list = []
    for i in range(0, 200, 10):
        rm = RandomForestClassifier(n_estimators=i + 1
                                    )
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

    # 二阶多项式决策边界，l1正则筛选重要关联特征
    # print("二阶多项式决策边界")
    # model = polynomial_model(degree=2, penalty='l1', solver='liblinear')
    # score = cross_val_score(model, X, Y, scoring='roc_auc', cv=scv, n_jobs=-1)
    # print("Cross Validation Scores are {}".format(score))
    # print("Average Cross Validation score :{}".format(score.mean()))


def score_model(x_fit_transform, data1, model, cv=None):
    x = x_fit_transform(data1.iloc[:, 6:])
    x = np.append(np.array(x[0].astype('float32')), np.array(x[1]), axis=1)
    X_train, X_test, y_train, y_test = train_test_split( \
        x, np.array(data1['label']), test_size=0.1, random_state=seed)

    pred_list = []
    truth_list = []
    res_list = []

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

    print('*' * 100)
    print("Cross Validation Scores are {}".format(score))
    print("Average Cross Validation score :{}".format(score.mean()))

    return pred_list, truth_list, res_list


def draw_res(data1, x_fit_transform, pred_list, truth_list, res_list):
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

    classes = ['0', '1']
    font_dict = dict(fontsize=30,
                     color='r',
                     family='Times New Roman',
                     weight='light',
                     style='italic',
                     )

    for y_test, predictions, auc in zip(truth_list, pred_list, res_list):

        cm = confusion_matrix(y_test, predictions)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
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


def data_split():
    df_data = pd.read_excel(args.data_file, sheet_name=args.sheet_name)
    # np.array(data1)[:, 5].astype('int')
    split = StratifiedShuffleSplit(n_splits=num_repeat, test_size=0.1)
    for train_index, test_index in split.split(df_data, np.array(df_data["OS(month)"]).astype('int')):
        df_train = df_data.loc[train_index]
        df_test = df_data.loc[test_index]
        # df_val = df_test
        split_train = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
        for temp_index, val_index in split_train.split(df_data, np.array(df_data["OS(month)"]).astype('int')):
            df_val = df_train.loc[train_index[val_index]]
            df_train = df_train.loc[train_index[temp_index]]
    x_train = x_fit_transform(df_train)
    x_val = x_transform(df_val)
    x_test = x_transform(df_test)
    x_data = x_transform(df_data)
    x_train = (x_train[0].astype('float32'), x_train[1])
    x_val = (x_val[0].astype('float32'), x_val[1])
    x_test = (x_test[0].astype('float32'), x_test[1])
    x_data = (x_data[0].astype('float32'), x_data[1])


def pca_aly():
    X = x_fit_transform(data1.drop('label', axis=1))
    X = np.append(np.array(X[0].astype('float32')), np.array(X[1]), axis=1)
    Y = np.array(data1['label'])

    # Y = np.array(data1)[:, 5].astype('int')
    # X = np.array(data1)[:, 6:]

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)  # 实例化
    pca = pca.fit(X)  # 拟合模型

    print(pca.explained_variance_ratio_)

    x_dr = pca.transform(X)  # 获取新矩阵
    y = Y
    # x_dr[y == 0, 0]  # 采用布尔索引

    # 画出分类图
    # plt.figure()  # 创建一个画布
    # plt.scatter(x_dr[y == 0, 0], x_dr[y == 0, 1], c="red", label='class 0')
    # plt.scatter(x_dr[y == 1, 0], x_dr[y == 1, 1], c="black", label='class 1')
    # # plt.scatter(x_dr[y==2,0],x_dr[y==2,1],c="orange",label = iris.target_names[2])
    # plt.legend()  # 显示图例
    # plt.title("PCA of medical dataset")  # 显示标题
    # plt.show()

    gbc = GradientBoostingClassifier(n_estimators=11)
    scv = StratifiedKFold(n_splits=18, random_state=0, shuffle=True)

    score = cross_val_score(gbc, x_dr, y, scoring='roc_auc', cv=scv, n_jobs=-1)
    print("Cross Validation Scores are {}".format(score))
    print("Average Cross Validation score :{}".format(score.mean()))

    # y_prob  = gbc.predict_proba(x_dr)[:, 1]

    # from sklearn.metrics import roc_curve, auc
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_prob)
    # roc_auc = auc(false_positive_rate, true_positive_rate)
    #
    # plt.figure(figsize=(10, 10))
    # plt.title('ROC')
    #
    # plt.plot(false_positive_rate,
    #          true_positive_rate,
    #          color='red',
    #          label='AUC = %0.2f' % roc_auc)
    #
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], linestyle='--')
    #
    # plt.axis('tight')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.show()


def tsne():
    X = x_fit_transform(data1.drop('label', axis=1))
    X = np.append(np.array(X[0].astype('float32')), np.array(X[1]), axis=1)
    Y = np.array(data1['label'])
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


if __name__ == '__main__':
    new_auc = []
    ipi_auc = []
    num_repeat = 1
    c_index_new = 0
    c_index_ipi = 0

    data1, target, feature = prepare_data()

    ft = data1.iloc[:, 6:]

    ft.rename(columns={'esophagectomy(0=open,1=minimally invasive)': 'esophagectomy',
                       'approach（ Ivor Lewis technique=0，McKeown technique=1）': 'approach',
                       'surgery（open Ivor Lewis=1，open McKeown=2，minimally invasive Ivor Lewis =3，minimally invasive McKeown=4）': 'surgery',
                       'gender(1=male,2=female)': 'gender',
                       'drinking(0=no,1=yes)': 'drinking',
                       'history(1=diabetes,2=hypertension,3=both)': 'history',
                       'surgery-history(0=no,1=yes)': 'surgery-history',
                       'family-history-of-cancer(0=no,1=yes)': 'fhc',
                       'pathology-type(1=SCC,2=ADC,3=others)': 'pt',
                       'otherorgans-involvement(0=no,1=yes)': 'oi',

                       },
              inplace=True)

    corr = ft.corr()
    plt.figure(figsize=(14, 14))
    # annot=True 显示每个方格的数据
    sns.heatmap(corr, annot=True)
    plt.show()

    feature_analysis(data1)

    # train_no_transform(data1)
    x_fit_transform, x_transform = feature_transform()

    # data_split()

    train_transform(data1, target, feature, x_fit_transform, x_transform)
    # pca_aly()

# _*_ coding:utf-8 _*_
"""
@Time     : 2022/7/4 11:23
@Author   : Wangxuanye
@File     : os_model_train.py
@Project  : esophagealcancer
@Software : PyCharm
@License  : (C)Copyright 2018-2028, Taogroup-NLPR-CASIA
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/7/4 11:23        1.0             None
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torchtuples as tt
from torch.autograd import Variable
from torchtuples.practical import MLPVanilla
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE, SMOTENC, SVMSMOTE

from pycox.models import LogisticHazard, CoxCC, DeepHit, MTLR
from pycox.models.loss import NLLLogistiHazardLoss
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from pycox.evaluation import EvalSurv
from pycox.utils import kaplan_meier

from sklearn.metrics import roc_auc_score, roc_curve, auc

from DeepSurvivalNet import NetAESurv, LossAELogHaz, Concordance

# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import optuna


def parse_args():
    parser = argparse.ArgumentParser(
        description='hyper parameters.'
    )
    parser.add_argument('--lr', type=float, default=0.0023841771341908524,
                        help="learning rate")  # 0.00021355517730720436 0009651371591901338 00048531337233286123 018581773783169907 022490203808460348 015246481310113314 0012248256978481585 0007248600867422703 008829323596828262
    parser.add_argument('--alpha', type=float, default=0.08891002965259283,
                        help="loss weight")  # 0.7769056354193079 0. 0267816165851851 8917607699911321 1000025257267704 012796427502039574 013946708948221212 4641325009173762 0281588270572475 16162722034791277 07111412677096003
    parser.add_argument('--batch_size', type=int, default=32)  # 32 16
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.2)  # 0.2
    parser.add_argument('--encoded_features', type=int, default=24)  # 16 32 12 20
    parser.add_argument('--duration', type=str, default="follow-up time", help="name of duration")
    parser.add_argument('--event', type=str, default="death（yes=1，no=0）", help="name of event")
    parser.add_argument('--data_file', type=str, default="data/esophagealcancer_multimodal_data.xlsx")
    parser.add_argument('--sheet_name', type=str, default="OS")
    parser.add_argument('--hidden_dim', type=int,
                        default=[95, 211, 225, 23])  # 199, 242, 225, 134   112, 16, 64, 48  40, 167, 45, 131
    # 118, 173, 152, 184  41, 192, 73, 220  92, 152, 162, 114 226, 230, 50, 163 65, 147, 240, 187 29, 150, 188, 59

    return parser.parse_args()


def train(cols_standardize, cols_binary, cols_categorical, df_data, df_train, df_val, df_test):
    # Best trial until now: multimodal score
    #  Value:  0.7698412698412699
    #  Params:
    #     batch_size: 16
    #     encoded_features: 4
    #     loss weight: 0.6337549922823906
    #     lr: 0.009631737047014479

    # Best trial until now: structured score
    #  Value:  0.7976190476190477
    #  Params:
    #     batch_size: 16
    #     encoded_features: 8
    #     loss weight: 0.9652061780263278
    #     lr: 0.007642223176122756

    seed = 12345
    np.random.seed(seed)
    random.seed(seed)
    _ = torch.manual_seed(seed)

    args = parse_args()
    '''
    Data loader
    '''
    # print(x_data)

    # print(len(df_train[df_train['OSR']==1]), len(df_train[df_train['OSR']==0]))
    # # sampling_strategy={0: len(df_train[df_train['OSR']==0]), 1: len(df_train[df_train['OSR']==1])},
    # sme = ADASYN(random_state=0)
    # Y = df_train['OSR']
    # X = df_train[['OS', 'Gender', 'ECOG', 'Ann Arbor', 'B symptoms', 'Extranodal involvement sites', 'MYC gene status', 'BCL2 gene status', 'BCL6 gene status', 'COO', 'Age', 'WBC', 'ANC', 'ALC', 'ALC/AMC', 'AMC', 'Hb', 'PLT', 'Albumin', 'LDH', 'CRP', 'β2M', 'Ki-67']]
    # X_res, y_res = sme.fit_resample(X, Y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    # # print(X_res.head())
    # df_train = X_res
    # df_train['OSR'] = y_res
    # df_data = pd.concat([df_test, df_val])
    # df_test = df_val.sample(frac=0.7)
    # df_val = df_val.drop(df_test.index)
    # print(100*"*")
    # print(df_train.head())
    # print(100*"*")

    '''
    Feature transform
    Standardize numerical variables, keep binary variables unchanged, and perform Entity Embedding on categorical variables
    '''
    # cols_standardize = ['AMC', 'PLT', 'LDH']
    # cols_binary = []
    # cols_categorical = ['Extranodal involvement sites', 'MYC gene status']

    standardize = [([col], StandardScaler()) for col in cols_standardize]
    binary = [(col, None) for col in cols_binary]
    categorical = [(col, OrderedCategoricalLong()) for col in cols_categorical]

    x_mapper_float = DataFrameMapper(standardize + binary)
    x_mapper_long = DataFrameMapper(categorical)

    x_fit_transform = lambda df: tt.tuplefy(x_mapper_float.fit_transform(df), x_mapper_long.fit_transform(df))
    x_transform = lambda df: tt.tuplefy(x_mapper_float.transform(df), x_mapper_long.transform(df))

    x_train = x_fit_transform(df_train)
    x_val = x_transform(df_val)
    x_test = x_transform(df_test)
    x_data = x_transform(df_data)
    x_train = (x_train[0].astype('float32'), x_train[1])
    x_val = (x_val[0].astype('float32'), x_val[1])
    x_test = (x_test[0].astype('float32'), x_test[1])
    x_data = (x_data[0].astype('float32'), x_data[1])

    '''
    Label transform
    '''
    num_durations = 10  # 网络输出时间点个数，离散化网格的大小（等距）
    # scheme = 'quantiles'
    labtrans = LogisticHazard.label_transform(num_durations)

    get_target = lambda df: (df[args.duration].values, df[args.event].values)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))
    durations_test, events_test = get_target(df_test)
    durations_train, events_train = get_target(df_train)
    durations_val, events_val = get_target(df_val)

    durations_all, events_all = get_target(df_data)

    print("train:{} test:{} val:{} all{}".format(x_train[0].shape, x_test[0].shape, x_val[0].shape, x_data[0].shape))

    y_train = (y_train[0], y_train[1].astype('float32'))
    y_val = (y_val[0], y_val[1].astype('float32'))

    train = tt.tuplefy(x_train, (y_train, x_train))
    val = tt.tuplefy(x_val, (y_val, x_val))

    '''
    hyper parameter
    '''
    num_embeddings = x_train[1].max(0) + 1
    embedding_dims = num_embeddings // 2
    in_features = x_train[0].shape[1]
    # print(in_features)
    encoded_features = args.encoded_features
    out_features = labtrans.out_features

    '''
    train
    '''
    net = NetAESurv(in_features, encoded_features, out_features, num_embeddings, embedding_dims, args.dropout,
                    args.hidden_dim)
    # print(net)
    loss = LossAELogHaz(args.alpha, num_embeddings, embedding_dims, dropout=0.2)  # 0.6
    optimizer = tt.optim.Adam(lr=args.lr, weight_decay=0)
    model = LogisticHazard(net, optimizer, duration_index=labtrans.cuts, loss=loss)

    # dl = model.make_dataloader(train, batch_size=12, shuffle=False)
    # batch = next(iter(dl))
    # # batch
    # model.compute_metrics(batch)
    # model.score_in_batches(*train)

    metrics = dict(
        loss_surv=LossAELogHaz(1, num_embeddings, embedding_dims, dropout=0.2),
        loss_ae=LossAELogHaz(0, num_embeddings, embedding_dims, dropout=0.2)
    )
    # metrics = None
    # callbacks = [tt.cb.EarlyStopping(), Concordance(x_val, durations_val, events_val, per_epoch=1)]
    callbacks = [tt.cb.EarlyStopping()]
    # callbacks = None

    log = model.fit(*train, args.batch_size, args.epochs, callbacks, verbose=False, val_data=val,
                    metrics=metrics)  # 模型训练
    # log.plot()
    res = model.log.to_pandas()
    # val_c_index = model.callbacks['Concordance'].to_pandas()
    # val_c_index.plot()
    # print(res.head())
    # print(100*"*")
    surv = model.interpolate(10).predict_surv_df(x_test)
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    test_c_index = ev.concordance_td('antolini')
    print("test c-index: ", test_c_index)

    surv_train = model.interpolate(10).predict_surv_df(x_train)
    ev_train = EvalSurv(surv_train, durations_train, events_train, censor_surv='km')
    c_index = ev_train.concordance_td('antolini')
    print("train c-index: ", c_index)
    #
    surv_val = model.interpolate(10).predict_surv_df(x_val)
    ev_val = EvalSurv(surv_val, durations_val, events_val, censor_surv='km')
    c_index = ev_val.concordance_td('antolini')
    print("val c-index: ", c_index)

    surv_all = model.interpolate(10).predict_surv_df(x_data)
    ev_all = EvalSurv(surv_all, durations_all, events_all, censor_surv='km')
    all_c_index = ev_all.concordance_td('antolini')
    print("all c-index: ", all_c_index)

    # print(model.predict(x_test))
    #
    # print("            0     1")
    # print("test data: ", str(len(np.where(events_test == 0)[0])) + "   ", len(np.where(events_test == 1)[0]))
    # print("train data:", str(len(np.where(events_train == 0)[0])) + "  ", len(np.where(events_train == 1)[0]))
    # print("tval data: ", str(len(np.where(events_val == 0)[0])) + "   ", len(np.where(events_val == 1)[0]))
    # time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    # # ev.brier_score(time_grid).plot()
    # # plt.ylabel('Brier score')
    # # _ = plt.xlabel('Time')
    # #
    # # ev.nbll(time_grid).plot()
    # # plt.ylabel('NBLL')
    # # _ = plt.xlabel('Time')
    #
    # print(ev.integrated_brier_score(time_grid))
    # #
    # print(ev.integrated_nbll(time_grid))

    # pre_result = model.interpolate(100).predict_surv_df(x_test)
    # time_index = np.array(list(pre_result.index))
    # event_true = np.array(df_test[[args.duration, args.event]].values.tolist())
    # return model.interpolate(100).predict_surv(x_test), time_index, event_true, all_c_index

    pre_result = model.interpolate(100).predict_surv_df(x_data)
    time_index = np.array(list(pre_result.index))
    event_true = np.array(df_data[[args.duration, args.event]].values.tolist())

    km_survial = kaplan_meier(*get_target(df_data))
    print(len(km_survial))

    return model.interpolate(100).predict_surv(x_data), time_index, event_true, test_c_index, pre_result, km_survial


def draw_ct(time_auc_score_structured, index_all, start, index, temp, result_structured):
    index_12, index_36, index_60 = index_all[0], index_all[1], index_all[2]
    plt.figure(num=1)
    plt.plot(index[start:], time_auc_score_structured)
    # plt.plot(index[start:], time_auc_score_multimodal)

    plt.legend(['structured score'])
    plt.ylabel('AUC Value')
    plt.xlabel('survival time (Months)')

    plt.text(12.1, 0.8, (12, "structured:" + str(format(time_auc_score_structured[index_12], '.4f')),
                         ), color='r')
    plt.text(36.1, 0.6, (36, "structured:" + str(format(time_auc_score_structured[index_36], '.4f')),
                         ), color='k')
    plt.text(60.1, 0.4, (60, "structured:" + str(format(time_auc_score_structured[index_60], '.4f')),
                         ), color='b')
    plt.vlines(12, 0.34, 0.9, colors="r", linestyles="dashed")
    plt.vlines(36, 0.34, 0.9, colors="k", linestyles="dashed")
    plt.vlines(60, 0.34, 0.9, colors="b", linestyles="dashed")

    all_index = [index_12, index_36, index_60]
    years = [1, 3, 5]
    for i in range(2, 5):
        ind = all_index[i - 2]
        fig = plt.figure(num=i)
        diagonal_x = [0, 1]
        diagonal_y = diagonal_x
        y_truth = 1 - temp[:, ind + start]

        ax_structured = fig.add_subplot(121)
        y_pred_structured = result_structured[:, ind + start]
        fpr, tpr, thersholds = roc_curve(y_truth, y_pred_structured, drop_intermediate=False)
        fit_fpr = np.polyfit(fpr, tpr, 7)
        # fit_fpr = np.polyfit(fpr, tpr, 4)
        fit_ploy = np.poly1d(fit_fpr)
        roc_auc = auc(fpr, tpr)
        tpr = fit_ploy(fpr)
        fpr = np.insert(fpr, 0, 0)
        tpr = np.insert(tpr, 0, 0)
        ax_structured.plot(fpr, tpr, 'r--')
        ax_structured.plot(diagonal_x, diagonal_y, 'k-')
        ax_structured.legend(['ROC (area = {0:.2f})'.format(roc_auc)])
        ax_structured.set_xlabel('False Positive Rate')
        ax_structured.set_ylabel('True Positive Rate')
        ax_structured.set_title('structured score ROC Curve (' + str(years[i - 2]) + ' year)')

        # ax_multimodal = fig.add_subplot(122)
        # y_pred_multimodal = result_multimodal[:, ind + start]
        # fpr, tpr, _ = roc_curve(y_truth, y_pred_multimodal)
        # fit_fpr = np.polyfit(fpr, tpr, 7)
        # fit_ploy = np.poly1d(fit_fpr)
        # roc_auc = auc(fpr, tpr)
        # tpr = fit_ploy(fpr)
        # fpr = np.insert(fpr, 0, 0)
        # tpr = np.insert(tpr, 0, 0)
        # ax_multimodal.plot(fpr, tpr, 'r--')
        # ax_multimodal.plot(diagonal_x, diagonal_y, 'k-')
        # ax_multimodal.legend(['ROC (area = {0:.2f})'.format(roc_auc)])
        # ax_multimodal.set_xlabel('False Positive Rate')
        # ax_multimodal.set_ylabel('True Positive Rate')
        # ax_multimodal.set_title('multimodal score ROC Curve (' + str(years[i - 2]) + ' year)')

    plt.show()


def draw_St(St_pd, km_survial):
    # plt.figure(num=1)
    # St_pd.plot(drawstyle='steps-post')
    # km_survial.plot(drawstyle='steps-post')

    print(St_pd)
    x = range(0, 97, 12)
    km_dt = [1, 0.893103448,
             0.813485699,
             0.747714515,
             0.702713178,
             0.681943331,
             0.564522625,
             0.30172761,
             0.150863805
             ]

    st_Structured = []
    st_Multimodal = []
    #
    target = 0
    for k in St_pd["Structured"].keys():
        if k % 12 < 1 and k - target < 1 and k - target > 0 and target <= 97:
            print(k)
            st_Structured.append(St_pd["Structured"][k])
            target += 12

    print(len(st_Structured))
    print(st_Structured)

    target = 0
    for k in St_pd["Multimodal"].keys():
        if k % 12 < 1 and k - target < 1 and k - target > 0 and target <= 97:
            st_Multimodal.append(St_pd["Multimodal"][k])
            target += 12

    print(len(st_Multimodal))
    print(st_Multimodal)

    # index = [12.017777777777777, 24.035555555555554, 36.053333333333335, 48.07111111111111, 60.08888888888889,
    #          72.10666666666667, ]

    plt.step(x, st_Structured)
    plt.step(x, st_Multimodal)
    plt.step(x, km_dt)
    plt.legend(['Structured', 'Multimodal', 'K-M'])
    plt.ylabel('S(t)')
    plt.xlabel('survival time (Months)')

    plt.show()


def draw(time_auc_score_structured, time_auc_score_multimodal, index_all, start, index, temp, result_structured,
         result_multimodal):
    index_12, index_36, index_60 = index_all[0], index_all[1], index_all[2]
    plt.figure(num=1)
    plt.plot(index[start:], time_auc_score_structured)
    plt.plot(index[start:], time_auc_score_multimodal)
    plt.legend(['Structured score', 'Multimodal score'])
    plt.ylabel('AUC Value')
    plt.xlabel('survival time (Months)')
    plt.text(24.1, 0.8, (24, "Struc:" + str(format(time_auc_score_structured[index_12], '.4f')),
                         "Multi:" + str(format(time_auc_score_multimodal[index_12], '.4f'))), color='r')
    plt.text(36.1, 0.6, (36, "Struc:" + str(format(time_auc_score_structured[index_36], '.4f')),
                         "Multi:" + str(format(time_auc_score_multimodal[index_36], '.4f'))), color='k')
    plt.text(60.1, 0.4, (60, "Struc:" + str(format(time_auc_score_structured[index_60], '.4f')),
                         "Multi:" + str(format(time_auc_score_multimodal[index_60], '.4f'))), color='b')
    plt.vlines(24, 0.34, 0.9, colors="r", linestyles="dashed")
    plt.vlines(36, 0.34, 0.9, colors="k", linestyles="dashed")
    plt.vlines(60, 0.34, 0.9, colors="b", linestyles="dashed")

    all_index = [index_12, index_36, index_60]
    years = [2, 3, 5]
    for i in range(2, 5):
        ind = all_index[i - 2]
        fig = plt.figure(num=i)
        diagonal_x = [0, 1]
        diagonal_y = diagonal_x
        y_truth = 1 - temp[:, ind + start]

        print(y_truth)
        ax_structured = fig.add_subplot(121)
        y_pred_structured = result_structured[:, ind + start]
        fpr, tpr, thersholds = roc_curve(y_truth, y_pred_structured, drop_intermediate=False)
        fit_fpr = np.polyfit(fpr, tpr, 7)
        fit_ploy = np.poly1d(fit_fpr)
        roc_auc = auc(fpr, tpr)
        tpr = fit_ploy(fpr)
        fpr = np.insert(fpr, 0, 0)
        tpr = np.insert(tpr, 0, 0)
        ax_structured.plot(fpr, tpr, 'r--')
        ax_structured.plot(diagonal_x, diagonal_y, 'k-')
        ax_structured.legend(['ROC (area = {0:.2f})'.format(roc_auc)])
        ax_structured.set_xlabel('False Positive Rate')
        ax_structured.set_ylabel('True Positive Rate')
        ax_structured.set_title('Struc score ROC Curve (' + str(years[i - 2]) + ' year)')

        ax_multimodal = fig.add_subplot(122)
        y_pred_multimodal = result_multimodal[:, ind + start]
        fpr, tpr, _ = roc_curve(y_truth, y_pred_multimodal)
        fit_fpr = np.polyfit(fpr, tpr, 7)
        fit_ploy = np.poly1d(fit_fpr)
        roc_auc = auc(fpr, tpr)
        tpr = fit_ploy(fpr)
        fpr = np.insert(fpr, 0, 0)
        tpr = np.insert(tpr, 0, 0)
        ax_multimodal.plot(fpr, tpr, 'r--')
        ax_multimodal.plot(diagonal_x, diagonal_y, 'k-')
        ax_multimodal.legend(['ROC (area = {0:.2f})'.format(roc_auc)])
        ax_multimodal.set_xlabel('False Positive Rate')
        ax_multimodal.set_ylabel('True Positive Rate')
        ax_multimodal.set_title('Multi score ROC Curve (' + str(years[i - 2]) + ' year)')

        plt.tight_layout()

    plt.show()


def T_checkout_LASSO(data):
    from scipy.stats import levene, ttest_ind
    from sklearn.linear_model import LassoCV

    classinformation = data["death（yes=1，no=0）"].unique()

    for temp_classinformation in classinformation:
        temp_data = data[data['death（yes=1，no=0）'].isin([temp_classinformation])]
        exec("df%s=temp_data" % temp_classinformation)

    df0 = data.loc[data["death（yes=1，no=0）"] == 0]
    df1 = data.loc[data["death（yes=1，no=0）"] == 1]

    print("death == 0 " + str(len(df0)))
    print("death == 1 " + str(len(df1)))

    counts = 0
    columns_index = []
    # for column_name in data.columns[25:]:
    # for column_name in data.columns[2:26]:
    for column_name in data.columns[2:]:
        print(column_name)
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

    print("LASSO降维：")
    print(index)
    return X


def main():
    structured_auc = []
    multimodal_auc = []
    num_repeat = 1
    c_index_structured = 0
    c_index_multimodal = 0
    # structured score
    test_index = []
    # test_start = 0
    # tt = 1
    args = parse_args()

    seed = 12345
    np.random.seed(seed)
    random.seed(seed)
    _ = torch.manual_seed(seed)

    df_data = pd.read_excel(args.data_file, sheet_name=args.sheet_name)

    # df_data = pd.read_csv(args.data_file)

    # df_data = df_data[["OS", "OSR", 'Extranodal involvement sites', 'MYC gene status', 'AMC', 'PLT', 'LDH']]
    # df_train = df_data
    # df_test = df_train.sample(frac=0.1)
    # df_train = df_train.drop(df_test.index)
    # df_val = df_train.sample(frac=0.1)
    # df_train = df_train.drop(df_val.index)

    # T_checkout_LASSO(df_data)
    # num_repeat
    split = StratifiedShuffleSplit(n_splits=num_repeat, test_size=0.1)
    for train_index, test_index in split.split(df_data, df_data["death（yes=1，no=0）"]):
        df_train = df_data.loc[train_index]
        df_test = df_data.loc[test_index]
        # df_val = df_test
        split_train = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
        for temp_index, val_index in split_train.split(df_train, df_train["death（yes=1，no=0）"]):
            df_val = df_train.loc[train_index[val_index]]
            df_train = df_train.loc[train_index[temp_index]]
        #
        # for _ in range(num_repeat):
        # cols_standardize = ['AMC', 'PLT', 'LDH']
        # cols_binary = []
        # cols_categorical = ['Extranodal involvement sites', 'MYC gene status']

        # cols_standardize = ['65', '231', '392', '438', '451', '522', '584', '641', '695', '723',
        #                     '725', '733', '767', '819', '971', '988']
        # cols_binary = []
        # cols_categorical = ['Nstages', 'gastric-lymph-node(0=no,1=yes)', 'intravascular-cancer-embolus(0=no,1=yes)',
        #                     'nerve-invasion(0=no,1=yes)', 'differentiation(1=low,2=medium,3=high)', ]

        # cols_standardize = ['age', 'smokingindex']
        # cols_binary = []
        # cols_categorical = ['drinking(0=no,1=yes)', 'paraesophageal-lymph-node(0=no,1=yes)',
        #                     'family-history-of-cancer(0=no,1=yes)', 'esophagectomy(0=open,1=minimally invasive)',
        #                     'approach（ Ivor Lewis technique=0，McKeown technique=1）', 'Tstages',
        #                     'differentiation(1=low,2=medium,3=high)']

        # cols_standardize = ['733']
        # cols_binary = []
        # cols_categorical = ['Nstages', 'gastric-lymph-node(0=no,1=yes)',
        #                     "right-recurrent-laryngeal-nerve-involvement(0=no,1=yes)", "carina-lymph-node(0=no,1=yes)",
        #                     "paraesophageal-lymph-node(0=no,1=yes)"]

        # cols_standardize = ["age"]
        # cols_binary = []
        # cols_categorical = ['Nstages', 'gastric-lymph-node(0=no,1=yes)',
        #                     "right-recurrent-laryngeal-nerve-involvement(0=no,1=yes)", "carina-lymph-node(0=no,1=yes)",
        #                     "paraesophageal-lymph-node(0=no,1=yes)", 'intravascular-cancer-embolus(0=no,1=yes)',
        #                     "left-recurrent-laryngeal-nerve-involvement(0=no,1=yes)"]

        cols_standardize = ["age"]
        cols_binary = []
        cols_categorical = ['Nstages', 'gastric-lymph-node(0=no,1=yes)',
                            "right-recurrent-laryngeal-nerve-involvement(0=no,1=yes)", "carina-lymph-node(0=no,1=yes)",
                            "paraesophageal-lymph-node(0=no,1=yes)", 'lymph-node(0=singel,1=multiple)',
                            "left-recurrent-laryngeal-nerve-involvement(0=no,1=yes)"]

        print(50 * "**")

        result_structured, index, event_test, c_index, pre_result_structured, km_survial = train(cols_standardize,
                                                                                                 cols_binary,
                                                                                                 cols_categorical,
                                                                                                 df_data,
                                                                                                 df_train, df_val,
                                                                                                 df_test)
        c_index_structured += c_index
        # if tt == 1:
        #     test_index = index
        #     tt = tt + 1
        # # print(index)
        # print((np.array(test_index) == index).all())

        # '''
        print(50 * "-*")
        # multimodal score
        # cols_standardize = ['Age', 'LDH']
        # cols_binary = []
        # cols_categorical = ['ECOG', 'Ann Arbor', 'Extranodal involvement sites']

        # cols_standardize = ['733']
        # cols_binary = []
        # cols_categorical = ['Nstages', 'gastric-lymph-node(0=no,1=yes)',
        #                     "right-recurrent-laryngeal-nerve-involvement(0=no,1=yes)", "carina-lymph-node(0=no,1=yes)",
        #                     "paraesophageal-lymph-node(0=no,1=yes)"]

        # cols_standardize = ['733', "age", "920"]
        # # cols_standardize = ["age"]
        # cols_binary = []
        # cols_categorical = ['Nstages', 'gastric-lymph-node(0=no,1=yes)',
        #                     "right-recurrent-laryngeal-nerve-involvement(0=no,1=yes)", "carina-lymph-node(0=no,1=yes)",
        #                     "paraesophageal-lymph-node(0=no,1=yes)", 'intravascular-cancer-embolus(0=no,1=yes)',
        #                     "left-recurrent-laryngeal-nerve-involvement(0=no,1=yes)"]

        # cols_standardize = ['65', '231', '392', '438', '451', '522', '584', '641', '695', '723',
        #                     '725', '733', '767', '819', '971', '988']
        # cols_binary = []
        # cols_categorical = ['Nstages', 'gastric-lymph-node(0=no,1=yes)', 'intravascular-cancer-embolus(0=no,1=yes)',
        #                     'nerve-invasion(0=no,1=yes)', 'differentiation(1=low,2=medium,3=high)', ]

        cols_standardize = ['733']
        cols_binary = []
        cols_categorical = ['Nstages', 'gastric-lymph-node(0=no,1=yes)',
                            "right-recurrent-laryngeal-nerve-involvement(0=no,1=yes)", "carina-lymph-node(0=no,1=yes)",
                            "paraesophageal-lymph-node(0=no,1=yes)", 'intravascular-cancer-embolus(0=no,1=yes)',
                            "left-recurrent-laryngeal-nerve-involvement(0=no,1=yes)"]


        result_multimodal, _, _, c_index, pre_result_multimodal, _ = train(cols_standardize, cols_binary,
                                                                           cols_categorical, df_data,
                                                                           df_train, df_val,
                                                                           df_test)
        c_index_multimodal += c_index

        temp = np.zeros([len(result_structured), len(index)])
        for i in range(len(result_structured)):
            # for i in range(result_structured.shape[1]):
            if event_test[i][1] == 0:
                continue
            temp[i][np.where(index >= event_test[i][0])] = 1
        # print(temp[:, 1:10])
        start = np.where(index >= min(event_test[:, 0]))[0][0] + 1
        # start = np.where(index >= min(event_test[:, 0]))[0][0]
        # print(len(event_test))
        # print(event_test)
        # print(min(event_test[:, 0]))
        # print("strat:" + str(start))
        # print(event_test[start - 1])

        time_auc_score_structured = []
        time_auc_score_multimodal = []
        # index_12 = np.where(index >= 12)[0][0] - start
        index_24 = np.where(index >= 24)[0][0] - start
        index_36 = np.where(index >= 36)[0][0] - start
        index_60 = np.where(index >= 60)[0][0] - start
        index_all = [index_24, index_36, index_60]

        for i in range(start, len(index)):
            y_truth = 1 - temp[:, i]
            y_pred_structured = result_structured[:, i]
            y_pred_multimodal = result_multimodal[:, i]
            auc_score = roc_auc_score(y_truth, y_pred_structured)
            time_auc_score_structured.append(auc_score)
            # ================================================== #
            auc_score = roc_auc_score(y_truth, y_pred_multimodal)
            time_auc_score_multimodal.append(auc_score)
        if len(structured_auc) == 0:
            structured_auc = time_auc_score_structured
            multimodal_auc = time_auc_score_multimodal
        else:
            continue
            # structured_auc = np.array(structured_auc) + np.array(time_auc_score_structured)
            # multimodal_auc = np.array(multimodal_auc) + np.array(time_auc_score_multimodal)
        print(100 * '*')
    structured_auc = np.array(structured_auc) / num_repeat
    multimodal_auc = np.array(multimodal_auc) / num_repeat
    print(100 * '*')
    print('structured score c-index:', c_index_structured / num_repeat)
    print('multimodal score c-index:', c_index_multimodal / num_repeat)

    st_structured = pre_result_structured.mean(axis=1)
    st_multimodal = pre_result_multimodal.mean(axis=1)

    # st_structured = []
    # st_multimodal = []
    #
    # for k in pre_result_structured.keys():
    #     np.where(pre_result_structured[k] > 0.5)

    St_pd = pd.DataFrame(columns=['Structured', 'Multimodal'], index=pre_result_structured.index)
    St_pd["Structured"] = st_structured
    St_pd["Multimodal"] = st_multimodal

    # St_pd["K-M"] = km_survial.values

    draw_St(St_pd, km_survial)
    draw(structured_auc, multimodal_auc, index_all, start, index, temp, result_structured, result_multimodal)
    # draw_ct(structured_auc, index_all, start, index, temp, result_structured)
    # '''


if __name__ == "__main__":
    main()
    # args = parse_args()
    # df_data = pd.read_excel(args.data_file, sheet_name=args.sheet_name)
    # # df_data = df_data[["OS", "OSR", 'Extranodal involvement sites', 'MYC gene status', 'AMC', 'PLT', 'LDH']]
    # df_train = df_data
    # df_test = df_train.sample(frac=0.1)
    # df_train = df_train.drop(df_test.index)
    # df_val = df_train.sample(frac=0.1)
    # df_train = df_train.drop(df_val.index)
    # print(len(df_train['OSR']))
    # print(df_train.index)
    #
    # sme = SMOTE(random_state=0)
    # Y = df_train['OSR']
    # X = df_train[['OS', 'Gender', 'ECOG', 'Ann Arbor', 'B symptoms', 'Extranodal involvement sites', 'MYC gene status', 'BCL2 gene status', 'BCL6 gene status', 'COO', 'Age', 'WBC', 'ANC', 'ALC', 'ALC/AMC', 'AMC', 'Hb', 'PLT', 'Albumin', 'LDH', 'CRP', 'β2M', 'Ki-67']]
    # X_res, y_res = sme.fit_resample(X, Y)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    # # print(X_res.head())
    # df_train = X_res
    # df_train['OSR'] = y_res
    # print(len(df_train['OSR']))
    # print(df_train.index)

    # structured score
    # cols_standardize = ['AMC', 'PLT', 'LDH']
    # cols_binary = []
    # cols_categorical = ['Extranodal involvement sites', 'MYC gene status']
    # # cols_standardize = ['WBC', 'ANC', 'Ki-67']
    # # cols_binary = []
    # # cols_categorical = ['BCL6 gene status', 'ECOG']
    # result_structured, index, event_test = main(cols_standardize, cols_binary, cols_categorical)
    #
    # # '''
    # print(50*"-*")
    # # multimodal score
    # cols_standardize = ['Age', 'LDH']
    # cols_binary = []
    # cols_categorical = ['ECOG', 'Ann Arbor', 'Extranodal involvement sites']
    # result_multimodal, _, _ = main(cols_standardize, cols_binary, cols_categorical)
    # plt.show()
    # print(result_structured)
    # print(50*"-*")
    # print(result_multimodal)
    # temp = np.zeros([len(result_structured), len(index)])
    # for i in range(len(result_structured)):
    #     if event_test[i][1] == 0:
    #         continue
    #     temp[i][np.where(index >= event_test[i][0])] = 1
    # # print(temp[:, 1:10])
    # start = np.where(index >= min(event_test[:, 0]))[0][0] + 1
    # time_auc_score_structured = []
    # time_auc_score_multimodal = []
    # index_12 = np.where(index >= 12)[0][0] - start
    # index_36 = np.where(index >= 36)[0][0] - start
    # index_60 = np.where(index >= 60)[0][0] - start
    #
    # for i in range(start, len(index)):
    #     y_truth = 1 - temp[:, i]
    #     y_pred_structured = result_structured[:, i]
    #     y_pred_multimodal = result_multimodal[:, i]
    #     auc_score = roc_auc_score(y_truth, y_pred_structured)
    #     time_auc_score_structured.append(auc_score)
    #     # ================================================== #
    #     auc_score = roc_auc_score(y_truth, y_pred_multimodal)
    #     time_auc_score_multimodal.append(auc_score)
    # print(time_auc_score_structured)
    # print(sum(time_auc_score_structured)/len(time_auc_score_structured))
    # plt.figure(num=1)
    # plt.plot(index[start:], time_auc_score_structured)
    # plt.plot(index[start:], time_auc_score_multimodal)
    # plt.legend(['structured score', 'multimodal score'])
    # plt.ylabel('AUC Value')
    # plt.xlabel('survival time (Months)')
    # plt.text(12.1, 0.8, (12, "structured:" + str(format(time_auc_score_structured[index_12], '.4f')), "multimodal:" + str(format(time_auc_score_multimodal[index_12], '.4f'))),color='r')
    # plt.text(36.1, 0.6, (36, "structured:" + str(format(time_auc_score_structured[index_36], '.4f')), "multimodal:" + str(format(time_auc_score_multimodal[index_36], '.4f'))),color='k')
    # plt.text(60.1, 0.4, (60, "structured:" + str(format(time_auc_score_structured[index_60], '.4f')), "multimodal:" + str(format(time_auc_score_multimodal[index_60], '.4f'))),color='b')
    # plt.vlines(12, 0.34, 0.9, colors="r", linestyles="dashed")
    # plt.vlines(36, 0.34, 0.9, colors="k", linestyles="dashed")
    # plt.vlines(60, 0.34, 0.9, colors="b", linestyles="dashed")
    #
    # all_index = [index_12, index_36, index_60]
    # years = [1, 3, 5]
    # for i in range(2, 5):
    #     ind = all_index[i-2]
    #     fig = plt.figure(num=i)
    #     diagonal_x = [0, 1]
    #     diagonal_y = diagonal_x
    #     y_truth = 1 - temp[:, ind + start]
    #
    #     ax_structured = fig.add_subplot(121)
    #     y_pred_structured = result_structured[:, ind + start]
    #     fpr, tpr, thersholds = roc_curve(y_truth, y_pred_structured)
    #     roc_auc = auc(fpr, tpr)
    #     ax_structured.plot(fpr, tpr, 'r--')
    #     ax_structured.plot(diagonal_x, diagonal_y, 'k-')
    #     ax_structured.legend(['ROC (area = {0:.2f})'.format(roc_auc)])
    #     ax_structured.set_xlabel('False Positive Rate')
    #     ax_structured.set_ylabel('True Positive Rate')
    #     ax_structured.set_title('structured score ROC Curve (' + str(years[i-2]) + ' year)')
    #
    #     ax_multimodal = fig.add_subplot(122)
    #     y_pred_multimodal = result_multimodal[:, ind + start]
    #     fpr, tpr, _ = roc_curve(y_truth, y_pred_multimodal)
    #     roc_auc = auc(fpr, tpr)
    #     ax_multimodal.plot(fpr, tpr, 'r--')
    #     ax_multimodal.plot(diagonal_x, diagonal_y, 'k-')
    #     ax_multimodal.legend(['ROC (area = {0:.2f})'.format(roc_auc)])
    #     ax_multimodal.set_xlabel('False Positive Rate')
    #     ax_multimodal.set_ylabel('True Positive Rate')
    #     ax_multimodal.set_title('multimodal score ROC Curve (' + str(years[i-2]) + ' year)')
    #
    # plt.show()
    # '''

# _*_ coding:utf-8 _*_
"""
@Time     : 2022/7/4 11:23
@Author   : Wangxuanye
@File     : os_model_train_adjust.py
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

from sklearn.metrics import roc_auc_score, roc_curve, auc

from DeepSurvivalNet import NetAESurv, LossAELogHaz, Concordance

# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import optuna
from optuna.trial import TrialState


def parse_args():
    parser = argparse.ArgumentParser(
        description='hyper parameters.'
    )
    parser.add_argument('--lr', type=float, default=0.00021355517730720436, help="learning rate")  #
    parser.add_argument('--alpha', type=float, default=0.8917607699911321, help="loss weight")  # 0.7769056354193079
    parser.add_argument('--batch_size', type=int, default=32)  # 32
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.2)  # 0.2
    parser.add_argument('--encoded_features', type=int, default=12)  # 16
    parser.add_argument('--duration', type=str, default="follow-up time", help="name of duration")
    parser.add_argument('--event', type=str, default="death（yes=1，no=0）", help="name of event")
    parser.add_argument('--data_file', type=str, default="data/esophagealcancer_multimodal_data.xlsx")
    parser.add_argument('--sheet_name', type=str, default="OS")
    parser.add_argument('--hidden_dim', type=int, default=[112, 16, 64, 48])

    return parser.parse_args()


def train(trial, cols_standardize, cols_binary, cols_categorical, df_data, df_train, df_val, df_test):
    # Best trial until now: IPI score
    #  Value:  0.7698412698412699
    #  Params:
    #     batch_size: 16
    #     encoded_features: 4
    #     loss weight: 0.6337549922823906
    #     lr: 0.009631737047014479

    # Best trial until now: new score
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

    param = {
        "lr": trial.suggest_loguniform("lr", 0.0002, 1e0),
        "alpha": trial.suggest_loguniform("alpha", 1e-2, 1e0),
        "batch_size": trial.suggest_int("batch_size", 16, 32, step=16),
        "encoded_features": trial.suggest_int("encoded_features", 8, 32, step=4),
        "hidden_dim[0]": trial.suggest_int("hidden_dim[0]", 16, 256),
        "hidden_dim[1]": trial.suggest_int("hidden_dim[1]", 16, 256),
        "hidden_dim[2]": trial.suggest_int("hidden_dim[2]", 16, 256),
        "hidden_dim[3]": trial.suggest_int("hidden_dim[3]", 16, 256),
    }

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
    encoded_features = param['encoded_features']
    out_features = labtrans.out_features
    hidden_dim = [param['hidden_dim[0]'], param['hidden_dim[1]'], param['hidden_dim[2]'],
                  param['hidden_dim[3]']]
    '''
    train
    '''
    net = NetAESurv(in_features, encoded_features, out_features, num_embeddings, embedding_dims, args.dropout,
                    hidden_dim)
    # print(net)
    loss = LossAELogHaz(param['alpha'], num_embeddings, embedding_dims, dropout=0.2)  # 0.6
    optimizer = tt.optim.Adam(lr=param['lr'], weight_decay=0)
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
    # print(encoded_features)
    log = model.fit(*train, param['batch_size'], args.epochs, callbacks, verbose=False, val_data=val,
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

    pre_result = model.interpolate(100).predict_surv_df(x_test)
    time_index = np.array(list(pre_result.index))
    event_true = np.array(df_test[[args.duration, args.event]].values.tolist())
    return model.interpolate(100).predict_surv(x_test), time_index, event_true, all_c_index


def main(trial):
    new_auc = []
    num_repeat = 1
    c_index_new = 0

    args = parse_args()

    seed = 12345
    np.random.seed(seed)
    random.seed(seed)
    _ = torch.manual_seed(seed)

    df_data = pd.read_excel(args.data_file, sheet_name=args.sheet_name)

    split = StratifiedShuffleSplit(n_splits=num_repeat, test_size=0.1)
    for train_index, test_index in split.split(df_data, df_data["death（yes=1，no=0）"]):
        df_train = df_data.loc[train_index]
        df_test = df_data.loc[test_index]
        # df_val = df_test
        split_train = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
        for temp_index, val_index in split_train.split(df_train, df_train["death（yes=1，no=0）"]):
            df_val = df_train.loc[train_index[val_index]]
            df_train = df_train.loc[train_index[temp_index]]

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

        cols_standardize = ['733']
        cols_binary = []
        cols_categorical = ['Nstages', 'gastric-lymph-node(0=no,1=yes)',
                            "right-recurrent-laryngeal-nerve-involvement(0=no,1=yes)", "carina-lymph-node(0=no,1=yes)",
                            "paraesophageal-lymph-node(0=no,1=yes)", 'intravascular-cancer-embolus(0=no,1=yes)',
                            "left-recurrent-laryngeal-nerve-involvement(0=no,1=yes)"]

        print(cols_standardize)
        print(cols_categorical)

        print(50 * "**")

        result_new, index, event_test, c_index = train(trial, cols_standardize, cols_binary, cols_categorical, df_data,
                                                       df_train, df_val, df_test)
        c_index_new += c_index

        temp = np.zeros([len(result_new), len(index)])
        for i in range(len(result_new)):
            if event_test[i][1] == 0:
                continue
            temp[i][np.where(index >= event_test[i][0])] = 1
        # print(temp[:, 1:10])
        start = np.where(index >= min(event_test[:, 0]))[0][0] + 1

        time_auc_score_new = []

        for i in range(start, len(index)):
            y_truth = 1 - temp[:, i]
            y_pred_new = result_new[:, i]

            auc_score = roc_auc_score(y_truth, y_pred_new)
            time_auc_score_new.append(auc_score)
            # ================================================== #

        if len(new_auc) == 0:
            new_auc = time_auc_score_new
        else:
            continue
            # new_auc = np.array(new_auc) + np.array(time_auc_score_new)
            # ipi_auc = np.array(ipi_auc) + np.array(time_auc_score_ipi)
        print(100 * '*')
    new_auc = np.array(new_auc) / num_repeat
    print(100 * '*')
    print(new_auc)
    print('new score c-index:', c_index_new / num_repeat)

    return c_index_new / num_repeat


if __name__ == "__main__":
    # main()
    study = optuna.create_study(direction="maximize")
    # , timeout = 1800
    study.optimize(main, n_trials=400)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

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

    # new score
    # cols_standardize = ['AMC', 'PLT', 'LDH']
    # cols_binary = []
    # cols_categorical = ['Extranodal involvement sites', 'MYC gene status']
    # # cols_standardize = ['WBC', 'ANC', 'Ki-67']
    # # cols_binary = []
    # # cols_categorical = ['BCL6 gene status', 'ECOG']
    # result_new, index, event_test = main(cols_standardize, cols_binary, cols_categorical)
    #
    # # '''
    # print(50*"-*")
    # # IPI score
    # cols_standardize = ['Age', 'LDH']
    # cols_binary = []
    # cols_categorical = ['ECOG', 'Ann Arbor', 'Extranodal involvement sites']
    # result_ipi, _, _ = main(cols_standardize, cols_binary, cols_categorical)
    # plt.show()
    # print(result_new)
    # print(50*"-*")
    # print(result_ipi)
    # temp = np.zeros([len(result_new), len(index)])
    # for i in range(len(result_new)):
    #     if event_test[i][1] == 0:
    #         continue
    #     temp[i][np.where(index >= event_test[i][0])] = 1
    # # print(temp[:, 1:10])
    # start = np.where(index >= min(event_test[:, 0]))[0][0] + 1
    # time_auc_score_new = []
    # time_auc_score_ipi = []
    # index_12 = np.where(index >= 12)[0][0] - start
    # index_36 = np.where(index >= 36)[0][0] - start
    # index_60 = np.where(index >= 60)[0][0] - start
    #
    # for i in range(start, len(index)):
    #     y_truth = 1 - temp[:, i]
    #     y_pred_new = result_new[:, i]
    #     y_pred_ipi = result_ipi[:, i]
    #     auc_score = roc_auc_score(y_truth, y_pred_new)
    #     time_auc_score_new.append(auc_score)
    #     # ================================================== #
    #     auc_score = roc_auc_score(y_truth, y_pred_ipi)
    #     time_auc_score_ipi.append(auc_score)
    # print(time_auc_score_new)
    # print(sum(time_auc_score_new)/len(time_auc_score_new))
    # plt.figure(num=1)
    # plt.plot(index[start:], time_auc_score_new)
    # plt.plot(index[start:], time_auc_score_ipi)
    # plt.legend(['New score', 'IPI score'])
    # plt.ylabel('AUC Value')
    # plt.xlabel('survival time (Months)')
    # plt.text(12.1, 0.8, (12, "New:" + str(format(time_auc_score_new[index_12], '.4f')), "IPI:" + str(format(time_auc_score_ipi[index_12], '.4f'))),color='r')
    # plt.text(36.1, 0.6, (36, "New:" + str(format(time_auc_score_new[index_36], '.4f')), "IPI:" + str(format(time_auc_score_ipi[index_36], '.4f'))),color='k')
    # plt.text(60.1, 0.4, (60, "New:" + str(format(time_auc_score_new[index_60], '.4f')), "IPI:" + str(format(time_auc_score_ipi[index_60], '.4f'))),color='b')
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
    #     ax_new = fig.add_subplot(121)
    #     y_pred_new = result_new[:, ind + start]
    #     fpr, tpr, thersholds = roc_curve(y_truth, y_pred_new)
    #     roc_auc = auc(fpr, tpr)
    #     ax_new.plot(fpr, tpr, 'r--')
    #     ax_new.plot(diagonal_x, diagonal_y, 'k-')
    #     ax_new.legend(['ROC (area = {0:.2f})'.format(roc_auc)])
    #     ax_new.set_xlabel('False Positive Rate')
    #     ax_new.set_ylabel('True Positive Rate')
    #     ax_new.set_title('New score ROC Curve (' + str(years[i-2]) + ' year)')
    #
    #     ax_ipi = fig.add_subplot(122)
    #     y_pred_ipi = result_ipi[:, ind + start]
    #     fpr, tpr, _ = roc_curve(y_truth, y_pred_ipi)
    #     roc_auc = auc(fpr, tpr)
    #     ax_ipi.plot(fpr, tpr, 'r--')
    #     ax_ipi.plot(diagonal_x, diagonal_y, 'k-')
    #     ax_ipi.legend(['ROC (area = {0:.2f})'.format(roc_auc)])
    #     ax_ipi.set_xlabel('False Positive Rate')
    #     ax_ipi.set_ylabel('True Positive Rate')
    #     ax_ipi.set_title('IPI score ROC Curve (' + str(years[i-2]) + ' year)')
    #
    # plt.show()
    # '''

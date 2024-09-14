import os
import time
import optuna
import pandas
from xpinyin import Pinyin
from net import disease_name
from utils.pack_model_kit import *
from torch.utils.data import Dataset
# 导入模型包
import xgboost
from pytorch_tabnet.tab_model import TabNetClassifier
from catboost import CatBoostClassifier, Pool

# from deepforest import CascadeForestClassifier

pinyin = Pinyin()
# 需要实现过采样和欠采样，如果要指定数据就在下面指定，如果没有则会根据样本的数量的倒数来直接平衡至1：1的状态
POSITIVE_WEIGHT = None
NEGATIVE_WEIGHT = None
MAX_EPOCH = 30  # 针对tabnet的训练轮次的参数


class SimpleDS(Dataset):
    def __init__(self, x):
        self.X = x

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return len(self.X)


def tabnet_train(trial, if_trial=True):
    """
    训练模型的主要函数，实现了划分数据集，拟合模型这两部分的内容，并且可以进行过采样和欠采样
    加载数据集还是太慢了
    """
    max_epoch = 100
    if if_trial:
        # 模型的参数
        n_d = trial.suggest_categorical('n_d', [8, 16, 32, 64])  # 一般取值8-64
        n_a = n_d
        n_steps = trial.suggest_int('n_steps', 3, 10)
        gamma = trial.suggest_float('gamma', 1, 2, step=0.1)  # 一般是1.0 - 2.0
        momentum = trial.suggest_float('momentum', 0.01, 0.4, step=0.01)  # 0.01 - 0.4
        n_shared = trial.suggest_int('n_shared', 1, 5)  # 1-5
        lr = trial.suggest_categorical("weightdecay", [1e-4, 1e-1, 1e-2, 1e-3])
        optimizer_params = dict(lr=lr)
    else:
        # 模型的参数，取参数搜索的最优结果
        n_steps = 6
        n_d = 8
        n_a = 8
        gamma = 1.4
        momentum = 0.01
        n_shared = 4
        optimizer_params = dict(lr=2e-2)

    begin = time.time()
    ds_pack = fetch_ds(None, POSITIVE_WEIGHT, NEGATIVE_WEIGHT)
    train_x, train_y = ds_pack['train']
    val_x, val_y = ds_pack['val']
    test_x, test_y = ds_pack['test']

    print(f'load dataset use time {time.time() - begin}')

    model = TabNetClassifier(n_d=n_d,
                             n_a=n_a,
                             gamma=gamma,
                             n_steps=n_steps,
                             momentum=momentum,
                             n_shared=n_shared,
                             optimizer_params=optimizer_params)
    model.fit(X_train=train_x,
              y_train=train_y,
              eval_set=[(train_x, train_y), (val_x, val_y)],
              eval_name=['train', 'valid'],
              eval_metric=['auc'],
              max_epochs=max_epoch)

    y_pred = model.predict(test_x)

    model_name = 'tabnet'
    prc_file_name = f'files/{trial.number}_{model_name}_prc.csv' if if_trial else f'files/{model_name}_prc.csv'
    metrics_file_name = f'files/{trial.number}_{model_name}_result.csv' if if_trial else f'files/{model_name}_result.csv'

    res = get_metrics(test_y, y_pred, prc_file_name)
    pandas.DataFrame([res]).to_csv(metrics_file_name)
    return res['prc_auc']


def catboost_train(trial: optuna.Trial, if_trial=True):
    """
    参数详见：https://catboost.ai/en/docs/references/training-parameters/
    """
    if if_trial:
        eta = trial.suggest_float('eta', 0.001, 0.2)
        depth = trial.suggest_int('depth', 3, 10)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 0, 20)
        max_ctr_complexity = trial.suggest_int('max_ctr_complexity', 2, 10)
        scale_pos_weight = trial.suggest_int('scale_pos_weight', 20, 30)

        params = {'eta': eta,
                  'depth': depth,
                  'min_data_in_leaf': min_data_in_leaf,
                  'max_ctr_complexity': max_ctr_complexity,
                  'scale_pos_weight': scale_pos_weight}

    else:
        params = {'eta': 0.04,
                  'depth': 3,
                  'min_data_in_leaf': 6,
                  'max_ctr_complexity': 10,
                  'scale_pos_weight': 30}
    X_train, y_train, X_val, y_val, X_test, y_test = split_my_dataset(*read_my_dataset())
    train_pool = Pool(data=X_train, label=y_train)
    test_pool = Pool(data=X_test, label=y_test)

    model = CatBoostClassifier(**params)
    model.fit(train_pool)

    y_pred = model.predict(test_pool)

    model_name = 'catboost'
    prc_file_name = f'files/{trial.number}_{model_name}_prc.csv' if if_trial else f'files/{model_name}_prc.csv'
    metrics_file_name = f'files/{trial.number}_{model_name}_result.csv' if if_trial else f'files/{model_name}_result.csv'

    res = get_metrics(y_test, y_pred, prc_file_name)
    pandas.DataFrame([res]).to_csv(metrics_file_name)
    return res['prc_auc']


def xgboost_train(trial: optuna.Trial, if_trial=True):
    """
    在参数搜索的过程中发现不同的booster需要的参数是不一样的，本函数默认使用Tree Booster
    参考：https://blog.csdn.net/u011707542/article/details/78522276
    参数列表见：https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters
    """
    if not if_trial:  # 最好的实验结果
        eta = 0.09  # 0.01 - 0.2
        max_depth = 4  # 3-10
        min_child_weight = 4  # 1-10
        gamma = 0.93  # 0-1
        scale_pos_weight = 30  # 10-30
    else:
        eta = trial.suggest_float('eta', 0.01, 0.2, step=0.01)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        gamma = trial.suggest_float('gamma', 0, 1)
        scale_pos_weight = trial.suggest_int('scale_pos_weight', 10, 30)

    X_train, y_train, X_val, y_val, X_test, y_test = split_my_dataset(*read_my_dataset())

    model = xgboost.XGBClassifier(
        eta=eta,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        scale_pos_weight=scale_pos_weight
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    model_name = 'xgboost'
    prc_file_name = f'files/{trial.number}_{model_name}_prc.csv' if if_trial else f'files/{model_name}_prc.csv'
    metrics_file_name = f'files/{trial.number}_{model_name}_result.csv' if if_trial else f'files/{model_name}_result.csv'

    res = get_metrics(y_test, y_pred, prc_file_name)
    pandas.DataFrame([res]).to_csv(metrics_file_name)
    return res['prc_auc']


def trial_task(study_name, study_func):
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # 数据文件存放的地址
        study_name=study_name,  # 需要指定学习任务的名字，该名字就是数据文件的名字
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(study_func, n_trials=40)
    print(f'best params:{study.best_params}, best value:{study.best_value}, best trial:{study.best_trial}')


# def gcforest_train():
#     ds_pack = fetch_ds(None, POSITIVE_WEIGHT, NEGATIVE_WEIGHT)
#     train_x, train_y = ds_pack['train']
#     test_x, test_y = ds_pack['test']
#
#     model = CascadeForestClassifier()
#     model.fit(train_x, train_y)
#
#     y_pred = model.predict(test_x)
#
#     res = get_metrics(test_y, y_pred, prc_save_path='gcforest_pr_auc.csv')
#     pandas.DataFrame([res]).to_csv('gcforest_res.csv')


def get_metrics(ture_y, pred_y, prc_save_path=None):
    from utils.pytorch_model_kit import TestMetricsRecoder

    my_metrics = TestMetricsRecoder()
    my_metrics.y_labels, my_metrics.y_outputs = ture_y, pred_y
    return my_metrics.get_metrics(prc_save_path=prc_save_path)


def walk_disease(file_suffix, model):
    """
    便利所有疾病数据集，并将结果储存为csv文件
    """
    result_list = []
    target_list = [None] + disease_name
    for dse in target_list:
        result_list.append(get_metrics(target_disease=dse))
    df = pandas.DataFrame(result_list)

    if file_suffix is None:
        df.to_csv(f'{model}.csv')
    else:
        df.to_csv(f'{model}_{file_suffix}.csv')


if __name__ == '__main__':
    tabnet_train(None, if_trial=False)
    catboost_train(None, if_trial=False)
    xgboost_train(None, if_trial=False)

# encoding=utf-8

import copy
import random
import numpy
import pandas
import torch
from torch.utils.data import Dataset
import multiprocessing as mul
import pickle

cluster_num = 6


def auto_mul_process(func, data_list, proc_num=None):
    if proc_num is None:
        if len(data_list) < 30:
            proc_num = len(data_list)
        else:
            proc_num = 30
    pool = mul.Pool(proc_num)
    rel = pool.map(func, data_list)
    pool.close()
    pool.join()
    return rel


def set_random_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(5)

# DATA_CSV_PATH = r"data/原始数据 KNN补全 Version4.csv"

DEATH_CSV_PATH = 'data/death_label.csv'
death_df = pandas.read_csv(DEATH_CSV_PATH, index_col=0)


def rand_fill_list(_list: list, _len):
    res_list = []
    while len(res_list) < _len:
        if _len - len(res_list) > len(_list):
            res_list.extend(random.sample(_list, len(_list)))
        else:
            res_list.extend(random.sample(_list, _len - len(res_list)))
    return res_list


def get_data_arr_dict():
    with open('data/data.pkl', 'rb') as f:
        res = pickle.load(f)
    return res


class GlobalDiseaseDataSet(Dataset):
    """
    全局任务模型使用的数据集，不用去区分病症
    """

    def __init__(self):
        self.data_arr_dict: dict = get_data_arr_dict()

        # 获取死亡的id
        self.death_id = death_df["idnum"].tolist()
        self.keys_list = list(self.data_arr_dict.keys())

    def __getitem__(self, index):
        id_num = self.keys_list[index]

        x = self.data_arr_dict[id_num].astype('float64')
        y = 1 if id_num in self.death_id else 0
        return x, y

    def __len__(self):
        return len(self.data_arr_dict)


class DiseaseDataSet(Dataset):
    """
    每个病症对应一个`
    返回每个病症的数据样本
    """

    def __init__(self):
        self.data_arr_dict = get_data_arr_dict()

        with open('data/gmm_6.pkl', 'rb') as f:
            self.idnum_2_cluster_dict: dict = pickle.load(f)

        # 获取死亡的id
        self.death_id = death_df["idnum"].tolist()

        # 获取聚类中心类别到患者的映射
        self.cluster_to_id_dict = {i: [] for i in range(cluster_num)}
        for item in self.idnum_2_cluster_dict.items():
            self.cluster_to_id_dict[item[1]].append(item[0])

        self.max_len = max(
            [len(self.cluster_to_id_dict[k]) for k in self.cluster_to_id_dict.keys()]
        )

        # 将所有的id列表补全至同一长度
        self.cluster_to_id_dict2 = copy.deepcopy(self.cluster_to_id_dict)
        for k in self.cluster_to_id_dict2.keys():
            self.cluster_to_id_dict2[k] = rand_fill_list(
                self.cluster_to_id_dict[k], self.max_len
            )

    def __getitem__(self, index):
        x = {
            cluster: self.data_arr_dict[self.cluster_to_id_dict2[cluster][index]]
            for cluster in range(cluster_num)
        }

        y = {cluster: 1
        if self.cluster_to_id_dict2[cluster][index] in self.death_id else 0
             for cluster in range(cluster_num)}

        for k in x.keys():
            x[k] = torch.Tensor(x[k].astype('float64'))
        return x, y

    def __len__(self):
        return self.max_len


if __name__ == "__main__":
    pass

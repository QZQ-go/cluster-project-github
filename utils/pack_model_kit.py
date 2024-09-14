"""
专门给有现成包的模型写的工具集合
"""
from typing import Union
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, WeightedRandomSampler, DataLoader
from data_loader import *


class CustomDataset(Dataset):
    def __init__(self, data_df, label):
        self.data_df: pandas.DataFrame = data_df
        self.label = label

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        return self.data_df.iloc[idx, :].to_numpy(), self.label[idx]


def read_my_dataset():
    """
    以读取csv文件的方式获取数据集，效率会更高
    """
    df = pandas.read_csv('myds.csv', index_col=0)
    df.reset_index(inplace=True, drop=True)
    label = '151'
    X = df.drop(label, axis=1).to_numpy()
    Y = df[label].to_numpy()
    return X, Y


def split_my_dataset(x, y):
    x2, X_test, y2, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(x2, y2, test_size=0.1, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test


def read_my_dataset_openml():
    """
    以openml的方式来读取数据，这个函数并不是给本工程用的，而是给别的包用的
    """
    df = pandas.read_csv('myds.csv', index_col=0)
    df.reset_index(inplace=True, drop=True)
    label = df['151']
    data_df = df.drop(columns=['151'])
    categorical_indicator = [False for _ in range(32)] + [True for _ in range(len(data_df.columns) - 32)]
    return data_df, label, categorical_indicator


def split_x_y_into_numpy(target_dataset: Union[DataLoader, Dataset]):
    """
    将数据集或者dataloader中的数据成对取出拼接成numpy的形式以便模型使用
    主要是考虑到一些模型只接受numpy形式的数据
    """
    x, y = [], []
    for i in target_dataset:
        x.append(numpy.array([i[0][0].numpy()]))
        y.append(numpy.array([i[1].numpy()]))
    x, y = tuple(x), tuple(y)
    x, y = numpy.concatenate(x), numpy.concatenate(y)
    return x, y


def get_over_load_sampler(data_set,
                          pos_weight,
                          neg_weight):
    """
    根据data_set的数据分布情况生成sampler以处理数据不平衡的情况
    """
    if pos_weight is None and neg_weight is None:
        # 如果没有指定权重系数，则需要自己来计算
        pos_num = sum([i[1] for i in data_set])
        neg_num = len(data_set) - pos_num
        pos_weight = 1 / (pos_num / len(data_set))
        neg_weight = 1 / (neg_num / len(data_set))

    weight_sequence = []
    for i in data_set:
        x, y = i[0], i[1]
        if y == 1:
            weight_sequence.append(pos_weight)
        else:
            weight_sequence.append(neg_weight)

    return WeightedRandomSampler(weights=weight_sequence,
                                 num_samples=len(data_set),
                                 replacement=True)


def convert_to_csv():
    """
    将格式稀巴烂的数据文件转化程至少可以用的程度
    简单来说就是将特征规整起来转化成csv文件，其中第151个特征是label特征（就是是否死亡）
    该文件一般作为输入文件给现成的包模型进行使用
    该包应该复制到主目录来使用
    """
    death_id = death_df["idnum"].tolist()
    print(len(death_id))
    data_dict = get_data_arr_dict()

    for k in data_dict.keys():
        if k in death_id:
            data_dict[k] = numpy.concatenate([data_dict[k], numpy.array([1])])
        else:
            data_dict[k] = numpy.concatenate([data_dict[k], numpy.array([0])])

    df = pandas.DataFrame(data_dict)
    df = df.T
    df.to_csv('myds.csv')
    df = pandas.read_csv('myds.csv')
    df.to_csv('myds1.csv', encoding='gbk')

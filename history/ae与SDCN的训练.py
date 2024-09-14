import pickle
import numpy
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import Linear, Parameter
from torch.optim import Adam
from DEC import target_distribution, eva

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
lr = 1e-3
n_clusters = 15
nz = 15


class AE(nn.Module):
    """
    定义了一个名为 AE(Autoencoder) 的自编码器类，继承自 nn.Module 类，以便可以使用 PyTorch 框架中的各种功能和特性
    三个编码器层 和 三个解码器层 每个层都由一个全连接线性层和rule激活函数组成

    关于数值的设定 参考了 https://blog.csdn.net/llismine/article/details/130655973
    """
    def __init__(self):
        super(AE, self).__init__()

        # encode 编码器
        self.enc_1 = Linear(151, 256)
        self.enc_2 = Linear(256, 256)
        self.enc_3 = Linear(256, 1024)

        self.z_layer = Linear(1024, nz)

        # decode 解码器
        self.dec_1 = Linear(nz, 1024)
        self.dec_2 = Linear(1024, 256)
        self.dec_3 = Linear(256, 256)

        self.x_bar_layer = Linear(256, 151)

    # 定义了前向传播函数
    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        # 最终输出
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))

        x_bar = self.x_bar_layer(dec_h3)
        # 将第三层解码器的输出作为输入，通过重构层进行线性转换得到重构数据 x_bar。
        return x_bar, enc_h1, enc_h2, enc_h3, z


class SDCN(nn.Module):
    # 具体地，SDCN 由两部分组成：自编码器和双重自监督模块
    # AE部分如上  双重自监督块则通过将数据点与聚类中心之间的距离转化为相似度，以双重自监督方式训练网络，从而实现聚类任务
    def __init__(self):
        super(SDCN, self).__init__()

        # AE部分
        # autoencoder for intra information
        self.ae = AE()

        # 将 cluster layer 定义为可学习Parameter变量
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, nz))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree  设置自编码器的度数参数 v
        self.v = 1

    def forward(self, x):
        # 前向传播过程，输入原始数据 x ，通过自编码器 ae 进行编解码操作，得到重构结果 x_bar 和编码表达 z 。
        # 同时，通过双重自监督模块计算样本点和聚类中心之间的相似度 q ，并将其标准化为概率分布形式，作为输出结果返回
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        # Dual Self-supervised Module  双重自监督模块
        # 这一步是软分配, 将点到聚类中心的距离转换为相似度
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)

        # 归一化：数据点属于某个聚类的概率  （一行加和为1）
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, z


def train_sdcn():
    # 定义 SDCN 模型，并初始化模型参数
    model = SDCN().to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    with open('data/数据字典/原始数据转array字典', 'rb') as f:
        res: dict = pickle.load(f)

    data = numpy.stack([i[1] for i in res.items()]).astype(float)
    ts_data = torch.tensor(data).float()

    with torch.no_grad():  # 利用自编码器对输入数据进行无监督学习，将其映射到低维空间中，并提取嵌入向量 z。
        _, _, _, _, z = model.ae(ts_data)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)

    y_pred = kmeans.fit_predict(z)  # 使用 KMeans 算法对嵌入向量进行聚类，并获取聚类结果。
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)  # 将聚类中心作为聚类层参数进行初始化。
    # 将 KMeans 聚类算法得到的聚类中心更新到 SDCN 模型的聚类层参数上。

    for epoch in range(200):  # 迭代200次
        # update_interval
        _, tmp_q, _ = model(ts_data)  # 对输入数据进行前向计算，得到模型的软分配概率 tmp_q
        tmp_q = tmp_q.data
        p = target_distribution(tmp_q)  # 调用函数 根据软分配概率 tmp_q 计算目标分布 p。
        # 软分配概率（n,k） （k为聚类中心个数）
        res1 = tmp_q.cpu().numpy().argmax(1)
        # Q  将软分配概率转换为 numpy 数组类型，并获取其中概率最大的索引作为聚类结果 res1

        res3 = p.data.cpu().numpy().argmax(1)
        # 计算并输出聚类结果的评价指标。
        eva(y_pred, res1, str(epoch) + 'Q')
        eva(y_pred, res3, str(epoch) + 'P')

        x_bar, q, _ = model(ts_data)  # 对输入数据进行前向计算，得到重构的输出数据 x_bar 和当前的软分配概率 q

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, ts_data)

        loss = kl_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model, 'ae.pkl')


if __name__ == '__main__':
    train_sdcn()

import torch.utils.data as Data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import random_split
from torch import nn
from torch.nn import Linear
from data_loader import GlobalDiseaseDataSet


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
lr = 1e-3
n_clusters = 15
nz = 10


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

        x_bar = F.sigmoid(self.x_bar_layer(dec_h3))
        # 将第三层解码器的输出作为输入，通过重构层进行线性转换得到重构数据 x_bar。
        return x_bar, enc_h1, enc_h2, enc_h3, z


ds = GlobalDiseaseDataSet()

train_data, test_data = random_split(ds, [0.8, 0.2])

train_loader = Data.DataLoader(dataset=train_data, batch_size=256, shuffle=True, num_workers=4)
test_loader = Data.DataLoader(dataset=test_data, batch_size=256, shuffle=True, num_workers=4)

model = AE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    epoch_loss = 0
    for batch_idx, (batch_x, _) in enumerate(train_loader):
        batch_x = batch_x.float()
        recon_x = model(batch_x.float())
        loss = F.mse_loss(recon_x[0].float(), batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.data.item()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(batch_x.cpu().detach()), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(batch_x.cpu().detach())
            ))

    epoch_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, epoch_loss))
    return epoch_loss


def model_test(epoch):
    model.eval()
    test_loss = 0
    feat_total = []
    target_total = []
    for i, (data, target) in enumerate(test_loader):
        data = data
        recon_batch, feat = model(data)
        test_loss += F.mse_loss((recon_batch, data.view(-1, 784))).item()
        feat_total.append(feat.data.cpu().view(-1, feat.data.shape[1]))
        target_total.append(target)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':
    for epoch in range(1, 21):
        train(epoch)
        torch.save(model.state_dict(), f'pretrain_ae_{epoch}.pkl')


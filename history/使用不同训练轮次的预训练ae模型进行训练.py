import argparse
import glob
import os.path
import torch.nn.functional as F
from torch.nn import Linear
from data_loader import *
from utils.pytorch_model_kit import *
from utils import init_console_and_file_log
from torch import torch, nn
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import StepLR

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
lr = 1e-3
n_clusters = 15
nz = 32
batch_size = 512


class AE(nn.Module):
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
train_parser = argparse.ArgumentParser(prog='train argparse')
sub_parse = train_parser.add_subparsers(title='subparsers')

# 模型训练需要的参数
mod_parser = sub_parse.add_parser(name='model',
                                  description='model params that needed to be saved')
mod_parser.add_argument('--epochs', type=int, default=60)
mod_parser.add_argument('--pos_weight', type=int, default=10,
                        help='a ratio adapted when ture sample was classified properly')
mod_parser.add_argument('--neg_weight', type=int, default=30,
                        help='a punishment ratio adapted when ture sample was classified improperly')
mod_parser.add_argument('--learning_rate', type=float, default=1e-2)
mod_parser.add_argument('--batch_size', type=int, default=256)
mod_parser.add_argument('--weight_decay', type=float, default=1e-4)

# dataloader类的参数
dl_parse = sub_parse.add_parser(name='dl')
dl_parse.add_argument('--if_pin_memory', type=bool, default=True)
dl_parse.add_argument('--num_work', type=int, default=4)
dl_parse.add_argument('--target_disease', type=str, default='all')
dl_parse.add_argument('--batch_size', type=int, default=mod_parser.parse_args().batch_size)

# 以下是各种文件保存的路径
path_parse = sub_parse.add_parser(name='path')
path_parse.add_argument('--prefix', type=str, default='global')
path_parse.add_argument('--dir_name', type=str, default='files/')

# 代码运行需要的一些参数
oth_parser = sub_parse.add_parser(name='other')
oth_parser.add_argument('--target_acc', type=float, default=0.6)
oth_parser.add_argument('--target_rec', type=float, default=0.6)


class TestModel(nn.Module):
    def __init__(self, ae_model_path):
        super().__init__()

        self.ae = AE()
        self.ae.load_state_dict(torch.load(ae_model_path))

        for param in self.ae.parameters():
            param.requires_grad = False  # 冻结整个ae层的参数

        self.f1 = nn.Linear(151, 256)
        self.f2 = nn.Linear(256, 64)
        self.f3 = nn.Linear(64, 1)

    def forward(self, x):
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        x = self.f1(x_bar)
        x = self.f2(x)
        x = F.sigmoid(self.f3(x))
        return x


def get_path_params_dict(_path_args):
    """
    该函数返回一个字典，其中包含所有需要保存的数据路径
    """
    dir_name, prefix = _path_args.dir_name, _path_args.prefix
    path_dict = {'logger_path': os.path.join(dir_name, f'{prefix}_train.log'),
                 'process_csv_path': os.path.join(dir_name, f'{prefix}_process_record.csv'),
                 'test_csv_path': os.path.join(dir_name, f'{prefix}_test_result.csv'),
                 'image_path': os.path.join(dir_name, f'{prefix}_task.png'),
                 'model_file_path': os.path.join(dir_name, f'{prefix}_model_para.pt'),
                 'indices_file_path': os.path.join(dir_name, f'{prefix}_test_indices.txt'),
                 'train_params': os.path.join(dir_name, f'{prefix}_train_params.csv'),
                 'prc_path': os.path.join(dir_name, f'{prefix}_prc.csv')}
    return path_dict


def save_params(name_space, csv_path='params.csv'):
    pandas.DataFrame([name_space.__dict__]).to_csv(csv_path)


def get_data_loader(_args):
    """
    获取dataloader形式的数据集
    """
    data_set = GlobalDiseaseDataSet()

    train_len = int(round(len(data_set) * 0.8, 0))
    test_len = int(round(len(data_set) * 0.1, 0))
    train_len += len(data_set) - train_len - test_len * 2
    train_dataset, val_dataset, test_dataset = random_split(
        dataset=data_set, lengths=[train_len, test_len, test_len], generator=torch.Generator().manual_seed(0)
    )

    data_set_dict = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    _data_loader_dict = {k: DataLoader(data_set_dict[k],
                                       shuffle=True,
                                       batch_size=_args.batch_size,
                                       num_workers=_args.num_work,
                                       pin_memory=_args.if_pin_memory) for k in data_set_dict.keys()}
    return _data_loader_dict


def global_model(_train_parser, _mod_parser, ae_model_path):
    # 解析各种传入的参数
    _mod_args = _mod_parser.parse_args()
    _path_args = _train_parser.parse_args(['path'])
    _oth_args = _train_parser.parse_args(['other'])
    _dl_args = _train_parser.parse_args(['dl'])
    _path_args.prefix = os.path.splitext(os.path.basename(ae_model_path))[0]  # 命名模型

    path_dict = get_path_params_dict(_path_args)  # 获取日志文件储存路径
    my_logger = init_console_and_file_log(_path_args.prefix, path_dict['logger_path'])  # 建立logger
    save_params(_mod_args, os.path.join(_path_args.dir_name, f'{_path_args.prefix}.csv'))  # 保存模型训练用参数

    data_loader_dict = get_data_loader(_dl_args)
    model = TestModel(ae_model_path)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=_mod_args.learning_rate,
                                momentum=0.09,
                                weight_decay=1e-4)

    my_trainer = PyTorchTrainer(model=model,
                                loss_func=nn.BCELoss(),
                                optimizer=optimizer,
                                scheduler=StepLR(optimizer, step_size=4, gamma=0.8),
                                train_data_loader=data_loader_dict['train'],
                                val_data_loader=data_loader_dict['val'],
                                test_data_loader=data_loader_dict['test'],
                                epoch=_mod_args.epochs,
                                procedure_csv_path=path_dict['process_csv_path'],
                                test_csv_path=path_dict['test_csv_path'],
                                image_path=path_dict['image_path'],
                                model_path=path_dict['model_file_path'],
                                prc_path=path_dict['prc_path'],
                                logger=my_logger,
                                neg_weight=_mod_args.neg_weight,
                                pos_weight=_mod_args.pos_weight)
    my_trainer.train()


if __name__ == '__main__':
    if not os.path.exists('files'):
        os.mkdir('files')
    for p in glob.glob('models/*'):
        global_model(train_parser, mod_parser, p)

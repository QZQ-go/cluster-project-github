import torch
import argparse

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

train_parser = argparse.ArgumentParser(prog='train argparse')
sub_parse = train_parser.add_subparsers(title='subparsers')

# 模型训练需要的参数
mod_parser = sub_parse.add_parser(name='model',
                                  description='model params that needed to be saved')
mod_parser.add_argument('--epochs', type=int, default=60)
mod_parser.add_argument('--pos_weight', type=int, default=30,
                        help='a ratio adapted when ture sample was classified properly')
mod_parser.add_argument('--neg_weight', type=int, default=10,
                        help='a punishment ratio adapted when ture sample was classified improperly')
mod_parser.add_argument('--learning_rate', type=float, default=1e-2)
mod_parser.add_argument('--batch_size', type=int, default=128)

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
import argparse
from pytorch_widedeep.models import SAINT, TabTransformer, FTTransformer
from data_loader import *
from net import *
from utils.pack_model_kit import CustomDataset
from utils.pytorch_model_kit import *
from utils import init_console_and_file_log
from torch import torch, nn
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import StepLR
import optuna

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
train_parser = argparse.ArgumentParser(prog='train argparse')
sub_parse = train_parser.add_subparsers(title='subparsers')

# 模型训练需要的参数
mod_parser = sub_parse.add_parser(name='model',
                                  description='model params that needed to be saved')
mod_parser.add_argument('--epochs', type=int, default=100)
mod_parser.add_argument('--pos_weight', type=int, default=30,
                        help='a ratio adapted when ture sample was classified properly')
mod_parser.add_argument('--neg_weight', type=int, default=10,
                        help='a punishment ratio adapted when ture sample was classified improperly')
mod_parser.add_argument('--learning_rate', type=float, default=1e-2)
mod_parser.add_argument('--batch_size', type=int, default=512)

# dataloader类的参数
dl_parse = sub_parse.add_parser(name='dl')
dl_parse.add_argument('--if_pin_memory', type=bool, default=True)
dl_parse.add_argument('--num_work', type=int, default=6)
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
    if _args.target_disease == 'all':
        data_set = GlobalDiseaseDataSet()
    else:
        data_set = SingleDiseaseDataSet(target_diseases=_args.target_disease)
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


def global_model(_train_parser, _mod_parser):
    # 解析各种传入的参数
    _mod_parser.add_argument('--weight_decay', type=float, default=1e-4)
    _mod_args = _mod_parser.parse_args()
    _path_args = _train_parser.parse_args(['path'])
    _oth_args = _train_parser.parse_args(['other'])
    _dl_args = _train_parser.parse_args(['dl'])

    _path_args.prefix = 'global'  # 命名模型
    path_dict = get_path_params_dict(_path_args)  # 获取日志文件储存路径
    my_logger = init_console_and_file_log(_path_args.prefix, path_dict['logger_path'])  # 建立logger
    save_params(_mod_args, os.path.join(_path_args.dir_name, f'{_path_args.prefix}.csv'))  # 保存模型训练用参数

    data_loader_dict = get_data_loader(_dl_args)
    model = GlobalTaskDnn()
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


def svm_model(_train_parser, _mod_parser):
    # 解析各种传入的参数
    _mod_parser.add_argument('--weight_decay', type=float, default=1e-4)
    _mod_args = _mod_parser.parse_args()
    _path_args = _train_parser.parse_args(['path'])
    _oth_args = _train_parser.parse_args(['other'])
    _dl_args = _train_parser.parse_args(['dl'])

    _path_args.prefix = 'svm'  # 命名模型
    path_dict = get_path_params_dict(_path_args)  # 获取日志文件储存路径
    my_logger = init_console_and_file_log(_path_args.prefix, path_dict['logger_path'])  # 建立logger
    save_params(_mod_args, os.path.join(_path_args.dir_name, f'{_path_args.prefix}.csv'))  # 保存模型训练用参数

    data_loader_dict = get_data_loader(_dl_args)
    model = SVM()  # 创建一个网络
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=_mod_args.learning_rate, momentum=0.9, dampening=0.1)  # 选择优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)  # 设置学习率下降策略

    my_trainer = PyTorchTrainerSVM(model=model,
                                   loss_func=None,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   train_data_loader=data_loader_dict['train'],
                                   val_data_loader=data_loader_dict['val'],
                                   test_data_loader=data_loader_dict['test'],
                                   epoch=_mod_args.epochs,
                                   procedure_csv_path=path_dict['process_csv_path'],
                                   test_csv_path=path_dict['test_csv_path'],
                                   image_path=path_dict['image_path'],
                                   model_path=path_dict['model_file_path'],
                                   logger=my_logger,
                                   neg_weight=_mod_args.neg_weight,
                                   pos_weight=_mod_args.pos_weight)
    my_trainer.train()


def logistic_model(_train_parser, _mod_parser):
    _mod_args = _mod_parser.parse_args()
    _path_args = _train_parser.parse_args(['path'])
    _oth_args = _train_parser.parse_args(['other'])
    _dl_args = _train_parser.parse_args(['dl'])

    _path_args.prefix = 'logistic'  # 命名模型
    path_dict = get_path_params_dict(_path_args)  # 获取日志文件储存路径
    my_logger = init_console_and_file_log(_path_args.prefix, path_dict['logger_path'])  # 建立logger
    save_params(_mod_args, os.path.join(_path_args.dir_name, f'{_path_args.prefix}.csv'))  # 保存模型训练用参数

    data_loader_dict = get_data_loader(_dl_args)
    model = LogisticRegression(151, 1)  # 创建一个网络
    model.initialize_weights()  # 初始化权值
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, dampening=0.1)  # 选择优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)  # 设置学习率下降策略

    my_trainer = PyTorchTrainer(model=model,
                                loss_func=nn.BCELoss(),
                                optimizer=optimizer,
                                scheduler=scheduler,
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


def cnn_1d_model(_train_parser, _mod_parser):
    # 解析各种传入的参数
    _mod_parser.add_argument('--weight_decay', type=float, default=1e-4)
    _mod_args = _mod_parser.parse_args()
    _path_args = _train_parser.parse_args(['path'])
    _oth_args = _train_parser.parse_args(['other'])
    _dl_args = _train_parser.parse_args(['dl'])

    _path_args.prefix = '1d_cnn'  # 命名模型
    path_dict = get_path_params_dict(_path_args)  # 获取日志文件储存路径
    my_logger = init_console_and_file_log(_path_args.prefix, path_dict['logger_path'])  # 建立logger
    save_params(_mod_args, os.path.join(_path_args.dir_name, f'{_path_args.prefix}.csv'))  # 保存模型训练用参数

    data_loader_dict = get_data_loader(_dl_args)
    model = MyCNN1D()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=_mod_args.learning_rate,
                                 weight_decay=_mod_args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=4,
                                                gamma=0.8)  # 设置学习率下降策略
    loss_func = nn.BCELoss()

    my_trainer = PyTorchTrainer(model=model,
                                loss_func=loss_func,
                                optimizer=optimizer,
                                scheduler=scheduler,
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


def pyt_wide_task(_train_parser, _mod_parser, use_model='saint'):
    _mod_args = _mod_parser.parse_args()
    _path_args = _train_parser.parse_args(['path'])
    _oth_args = _train_parser.parse_args(['other'])
    _dl_args = _train_parser.parse_args(['dl'])

    _path_args.prefix = 'saint'  # 命名模型
    path_dict = get_path_params_dict(_path_args)  # 获取日志文件储存路径
    my_logger = init_console_and_file_log(_path_args.prefix, path_dict['logger_path'])  # 建立logger
    save_params(_mod_args, os.path.join(_path_args.dir_name, f'{_path_args.prefix}.csv'))  # 保存模型训练用参数

    # 读取数据集，获取数据集特征
    df = pandas.read_csv('myds.csv', index_col=0)
    df.reset_index(inplace=True, drop=True)
    label = df['151']
    data_df = df.drop(columns=['151'])

    continuous_cols = [str(i) for i in range(0, 32)]
    cat_dims = [len(df.iloc[:, i].unique()) for i in range(32, 151)]
    cat_embed_input = [(str(u), i) for u, i in zip(range(32, 151), cat_dims)]
    colnames = [str(i) for i in range(151)]
    column_idx = {k: v for v, k in enumerate(colnames)}

    # 构建数据集
    data_set = CustomDataset(data_df, label)
    train_len = int(round(len(data_set) * 0.8, 0))
    test_len = int(round(len(data_set) * 0.1, 0))
    train_len += len(data_set) - train_len - test_len * 2
    train_dataset, val_dataset, test_dataset = random_split(
        dataset=data_set, lengths=[train_len, test_len, test_len], generator=torch.Generator().manual_seed(0)
    )

    data_set_dict = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    _data_loader_dict = {k: DataLoader(data_set_dict[k],
                                       shuffle=True,
                                       batch_size=_dl_args.batch_size,
                                       num_workers=_dl_args.num_work,
                                       pin_memory=_dl_args.if_pin_memory) for k in data_set_dict.keys()}

    if use_model == 'saint':
        model = SAINT(column_idx=column_idx,
                      cat_embed_input=cat_embed_input,
                      continuous_cols=continuous_cols,
                      mlp_hidden_dims=[64, 1])
    elif use_model == 'tab':
        model = nn.Sequential(TabTransformer(column_idx=column_idx,
                                             cat_embed_input=cat_embed_input,
                                             continuous_cols=continuous_cols,
                                             mlp_hidden_dims=[64, 1]), nn.Sigmoid())
    else:
        model = nn.Sequential(FTTransformer(column_idx=column_idx,
                                            cat_embed_input=cat_embed_input,
                                            continuous_cols=continuous_cols,
                                            mlp_hidden_dims=[64, 1]), nn.Sigmoid())
    model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=_mod_args.learning_rate, momentum=0.9, dampening=0.1)  # 选择优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)  # 设置学习率下降策略

    my_trainer = PyTorchTrainer(model=model,
                                loss_func=criterion,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                train_data_loader=_data_loader_dict['train'],
                                val_data_loader=_data_loader_dict['val'],
                                test_data_loader=_data_loader_dict['test'],
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
    # pyt_wide_task(train_parser, mod_parser, 'fft')
    global_model(train_parser, mod_parser)

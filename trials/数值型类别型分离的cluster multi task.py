import copy
import shutil
import pandas
import os
import matplotlib.pyplot as plt
import torch
import optuna
from collections import OrderedDict
from torch import nn
from data_loader import DiseaseDataSet
from torch.utils.data import random_split, DataLoader
from utils import init_console_and_file_log
from tqdm import tqdm
from utils.pytorch_model_kit import TestMetricsRecoder, MetricsRecoder
from config import device
from itertools import product

torch.multiprocessing.set_sharing_strategy('file_system')
cluster_num = 6


class MultiTaskDnn(nn.Module):
    def __init__(self,
                 num_layer=3,
                 cat_layer=3,
                 center_layer=1,
                 head_body=1):
        super().__init__()

        # 三种num_layer
        if num_layer == 3:
            self.num_body = nn.Sequential(
                OrderedDict(
                    [
                        ("linear1", nn.Linear(32, 128)),
                        ("relu1", nn.ReLU()),
                        ("dropout1", nn.Dropout(0.2)),

                        ("linear2", nn.Linear(128, 64)),
                        ("relu2", nn.ReLU()),
                        ("dropout2", nn.Dropout(0.2)),

                        ("linear3", nn.Linear(64, 32)),
                        ("relu3", nn.ReLU()),
                        ("dropout3", nn.Dropout(0.2)),
                    ]
                )
            )
        elif num_layer == 4:
            self.num_body = nn.Sequential(
                OrderedDict(
                    [
                        ("linear1", nn.Linear(32, 256)),
                        ("relu1", nn.ReLU()),
                        ("dropout1", nn.Dropout(0.2)),

                        ("linear2", nn.Linear(256, 128)),
                        ("relu2", nn.ReLU()),
                        ("dropout2", nn.Dropout(0.2)),

                        ("linear3", nn.Linear(128, 64)),
                        ("relu3", nn.ReLU()),
                        ("dropout3", nn.Dropout(0.2)),

                        ("linear4", nn.Linear(64, 32)),
                        ("relu4", nn.ReLU()),
                        ("dropout4", nn.Dropout(0.2)),
                    ]
                )
            )
        elif num_layer == 2:
            self.num_body = nn.Sequential(
                OrderedDict(
                    [
                        ("linear1", nn.Linear(32, 64)),
                        ("relu1", nn.ReLU()),
                        ("dropout1", nn.Dropout(0.2)),

                        ("linear3", nn.Linear(64, 32)),
                        ("relu3", nn.ReLU()),
                        ("dropout3", nn.Dropout(0.2)),
                    ]
                )
            )

        # 三种num_layer
        if cat_layer == 3:
            self.cat_body = nn.Sequential(
                OrderedDict(
                    [
                        ("linear1", nn.Linear(119, 256)),
                        ("relu1", nn.ReLU()),
                        ("linear2", nn.Linear(256, 128)),
                        ("relu2", nn.ReLU()),
                        ("linear3", nn.Linear(128, 64)),
                        ("relu3", nn.ReLU()),
                    ]
                )
            )
        elif cat_layer == 4:
            self.cat_body = nn.Sequential(
                OrderedDict(
                    [
                        ("linear1", nn.Linear(119, 512)),
                        ("relu1", nn.ReLU()),
                        ("linear2", nn.Linear(512, 256)),
                        ("relu2", nn.ReLU()),
                        ("linear3", nn.Linear(256, 128)),
                        ("relu3", nn.ReLU()),
                        ("linear4", nn.Linear(128, 64)),
                        ("relu4", nn.ReLU()),
                    ]
                )
            )
        elif cat_layer == 2:
            self.cat_body = nn.Sequential(
                OrderedDict(
                    [
                        ("linear1", nn.Linear(119, 128)),
                        ("relu1", nn.ReLU()),
                        ("linear2", nn.Linear(128, 64)),
                        ("relu2", nn.ReLU()),
                    ]
                )
            )

        if center_layer == 1:
            self.center_body = nn.Sequential(
                OrderedDict(
                    [
                        ("linear1", nn.Linear(96, 64)),
                        ("relu1", nn.ReLU()),
                    ]
                )
            )
        elif center_layer == 2:
            self.center_body = nn.Sequential(
                OrderedDict(
                    [
                        ("linear1", nn.Linear(96, 128)),
                        ("relu1", nn.ReLU()),
                        ("dropout1", nn.Dropout(0.2)),

                        ("linear2", nn.Linear(128, 64)),
                        ("relu2", nn.ReLU()),
                        ("dropout2", nn.Dropout(0.2)),
                    ]
                )
            )
        elif center_layer == 3:
            self.center_body = nn.Sequential(
                OrderedDict(
                    [
                        ("linear1", nn.Linear(96, 256)),
                        ("relu1", nn.ReLU()),
                        ("dropout1", nn.Dropout(0.2)),
                        ("linear2", nn.Linear(256, 128)),
                        ("relu2", nn.ReLU()),
                        ("dropout2", nn.Dropout(0.2)),
                        ("linear3", nn.Linear(128, 64)),
                        ("relu3", nn.ReLU()),
                        ("dropout3", nn.Dropout(0.2)),
                    ]
                )
            )

        if head_body == 1:
            self.head_pattern = nn.Sequential(
                OrderedDict(
                    [
                        ("linear2", nn.Linear(64, 1)),
                        ("Sigmoid", nn.Sigmoid()),
                    ]
                )
            )
        elif head_body == 2:
            self.head_pattern = nn.Sequential(
                OrderedDict(
                    [
                        ("linear1", nn.Linear(64, 64)),
                        ("relu1", nn.ReLU()),
                        ("dropout1", nn.Dropout(0.2)),
                        ("linear2", nn.Linear(64, 1)),
                        ("Sigmoid", nn.Sigmoid()),
                    ]
                )
            )

        self.multi_head_dict = nn.ModuleDict(
            {str(clst): copy.deepcopy(self.head_pattern) for clst in range(cluster_num)})

    def forward(self, x):
        """
        :param x: 由于需要进行分头训练，所以数据是按照字典的方式储存的，需要转译一下
        :return:
        """
        head_res = {}
        for clst in range(cluster_num):
            _x = x[clst]
            num_tes = _x[:, :32]
            cat_tes = _x[:, 32:]
            x_num = self.num_body(num_tes)
            x_tes = self.cat_body(cat_tes)
            _x = torch.concat([x_num, x_tes], dim=1)
            _x = self.center_body(_x)
            _x = self.multi_head_dict[str(clst)](_x)
            head_res.update({clst: _x})
        return head_res


def my_task(trial: optuna.trial.Trial):
    num_layer = trial.suggest_categorical('num_layer', [2, 3, 4])
    cat_layer = trial.suggest_categorical('cat_layer', [2, 3, 4])
    center_layer = trial.suggest_categorical('center_layer', [1, 2, 3])
    head_body = trial.suggest_categorical('head_body', [1, 2])

    BATCH_SIZE = 512
    POS_WEIGHT = 30
    NEG_WEIGHT = 10
    LR = 0.06
    momentum = 0.01
    weight_decay = 1.1e-4
    step_size = 8
    gamma = 0.7
    random_seed = 5

    EPOCHS = 120
    AIMED_ACC = 0.7
    AIMED_REC = 0.7
    NUM_WORK = 0
    PIN_MEMORY = False

    PREFIX = f'{trial.number}_mt_dnn'

    if not os.path.exists("trial"):
        os.mkdir('trial')
    target_dir = f'trial/trial_{trial.number}'
    os.mkdir(target_dir)

    LOGGER_FILE_NAME = os.path.join(target_dir, f'{PREFIX}_train.log')
    MODEL_FILE_NAME = os.path.join(target_dir, f'{PREFIX}_model_para.pt')
    IMAGE_NAME = os.path.join(target_dir, f'{PREFIX}_task.png')
    CSV_NAME = os.path.join(target_dir, f'{PREFIX}_task.csv')
    SINGLE_DISEASE_CSV_NAME = os.path.join(target_dir, f'{PREFIX}_task_single_disease.csv')
    PR_AUC_FILE_NAME = os.path.join(target_dir, f'{PREFIX}_pr_auc.csv')
    METRICS_PATH = os.path.join(target_dir, f'{PREFIX}_metrics.csv')
    SINGLE_DISEASE_METRICS_PATH = os.path.join(target_dir, f'{PREFIX}_single_disease_metrics.csv')

    logger = init_console_and_file_log("Trainer", LOGGER_FILE_NAME)
    logger.info(f'use device {device.type}')

    recorder = {f'{tp}_{mtc}': [] for tp, mtc in product(['train', 'val'], ['loss', 'rec', 'acc'])}
    clst_recorder = {f'{des}_{tp}_{mtc}': [] for des, tp, mtc in
                     product(range(cluster_num), ['train', 'val'], ['loss', 'rec', 'acc'])}

    best_acc = 0
    best_rec = 0
    best_value = 0

    # 定义数据集
    data_set = DiseaseDataSet()

    train_len = int(round(len(data_set) * 0.8, 0))
    test_len = int(round(len(data_set) * 0.1, 0))
    train_len += len(data_set) - train_len - test_len * 2

    train_dataset, val_dataset, test_dataset = random_split(
        dataset=data_set,
        lengths=[train_len, test_len, test_len],
        generator=torch.Generator().manual_seed(random_seed)
    )

    data_set_dict = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    data_loader_dict = {k: DataLoader(data_set_dict[k],
                                      shuffle=True,
                                      batch_size=BATCH_SIZE,
                                      num_workers=NUM_WORK,
                                      pin_memory=PIN_MEMORY) for k in data_set_dict.keys()}

    # 定义模型等
    model = MultiTaskDnn(num_layer, cat_layer, center_layer, head_body)
    model.to(device)

    # 定义损失函数等
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=LR,
                                momentum=momentum,
                                weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=step_size,
                                                gamma=gamma)

    for epoch in range(EPOCHS):
        logger.info(f"epoch {epoch}")
        torch.cuda.empty_cache()  # 释放显存

        for mode in ["train", "val"]:
            model.train() if mode == "train" else model.eval()

            t_metrics = MetricsRecoder()
            dse_t_metrics = {clst: MetricsRecoder() for clst in range(cluster_num)}

            target_data_loader = data_loader_dict[mode]
            for fold_num, data in enumerate(tqdm(target_data_loader)):
                if mode == "train":
                    optimizer.zero_grad()

                label: dict = data[1]
                inputs: dict = data[0]
                for k in inputs.keys():
                    inputs[k] = inputs[k].to(device)
                    label[k] = label[k].float().to(device)

                outputs: dict = model(inputs)
                for k in outputs.keys():
                    outputs[k] = outputs[k].view(outputs[k].shape[0])

                loss_list = []
                for k in label.keys():
                    # 计算loss
                    # 第二种权重赋值方式，当是正例并且是判断错误的时候才增加权重
                    weight_tensor = torch.where(
                        (label[k] == 1) & (outputs[k].round() == 0), POS_WEIGHT, 1
                    )
                    weight_tensor = torch.where(
                        (label[k] == 1) & (outputs[k].round() == 1), NEG_WEIGHT, weight_tensor
                    )

                    loss_func.weight = weight_tensor.to(device)
                    loss = loss_func(outputs[k], label[k])
                    loss_list.append(loss)

                    t_metrics.load(label[k], outputs[k], loss)
                    dse_t_metrics[k].load(label[k], outputs[k], loss)

                final_loss: torch.Tensor = sum(loss_list)

                if mode == "train":
                    final_loss.backward()
                    optimizer.step()

            # 记录该轮训练产生的数据
            acc, rec, total_loss = t_metrics.get_metrics()
            dse_res_dict = {clst: dse_t_metrics[clst].get_metrics() for clst in range(cluster_num)}

            recorder[f'{mode}_loss'].append(total_loss)
            recorder[f'{mode}_rec'].append(rec)
            recorder[f'{mode}_acc'].append(acc)

            logger.info(f"{mode} loss {total_loss}")
            logger.info(f"{mode} acc {acc}")
            logger.info(f"{mode} rec {rec}")

            for clst in range(cluster_num):
                clst_recorder[f'{clst}_{mode}_acc'].append(dse_res_dict[clst][0])
                clst_recorder[f'{clst}_{mode}_rec'].append(dse_res_dict[clst][1])
                clst_recorder[f'{clst}_{mode}_loss'].append(dse_res_dict[clst][2])

        scheduler.step()

        # 每个世代对结果进行制图
        fig = plt.figure()
        # 在第1，2，4的位置添加面板
        ax_loss, ax_acc, ax_rec = fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(224)

        ax_loss.set(title='Loss')
        ax_acc.set(title='Acc')
        ax_rec.set(title='Rec')

        for _type in ["train", "val"]:
            ax_loss.plot(recorder[f'{_type}_loss'], label=_type)
            ax_acc.plot(recorder[f'{_type}_acc'], label=_type)
            ax_rec.plot(recorder[f'{_type}_rec'], label=_type)

        plt.legend()
        plt.savefig(IMAGE_NAME)

        # 将训练记录保存为一个csv文件
        df = pandas.DataFrame(recorder)
        df.to_csv(CSV_NAME)
        df = pandas.DataFrame(clst_recorder)
        df.to_csv(SINGLE_DISEASE_CSV_NAME)

        # 如果精确度和召回率达到预期，则开始储存模型
        val_acc = recorder[f'val_acc'][-1]
        val_rec = recorder[f'val_rec'][-1]

        if val_acc > AIMED_ACC or val_rec > AIMED_REC:
            if val_acc > best_acc or val_rec > best_rec:
                best_acc = val_acc
                best_rec = val_rec
                torch.save(model.state_dict(), MODEL_FILE_NAME)

        if epoch % 4 == 0 and os.path.exists(MODEL_FILE_NAME):
            logger.info('start testing!')  # 当模型训练结束，加载最优参数进行结果测试
            test_model = MultiTaskDnn(num_layer, cat_layer, center_layer, head_body)
            test_model.load_state_dict(torch.load(MODEL_FILE_NAME))
            test_model.to(device)
            test_model.eval()

            metrics_dict = {str(clst): TestMetricsRecoder() for clst in range(cluster_num)}
            metrics_dict.update({'all': TestMetricsRecoder()})

            for fold_num, data in enumerate(tqdm(data_loader_dict['test'])):
                label: dict = data[1]
                inputs: dict = data[0]
                for k in inputs.keys():
                    inputs[k] = inputs[k].to(device)
                    label[k] = label[k].float().to(device)

                outputs: dict = test_model(inputs)

                for k in label.keys():
                    metrics_dict['all'].load(label[k], outputs[k], None)
                    metrics_dict[str(k)].load(label[k], outputs[k], None)

            all_metrics = metrics_dict['all'].get_metrics(PR_AUC_FILE_NAME)
            pandas.DataFrame([all_metrics]).T.to_csv(METRICS_PATH)
            metrics_dict.pop('all')
            res = {key: metrics_dict[key].get_metrics(os.path.join(target_dir, f'{PREFIX}_{key}_pr_auc.csv')) for key in
                   metrics_dict.keys()}
            pandas.DataFrame(res).to_csv(SINGLE_DISEASE_METRICS_PATH)

            if all_metrics['prc_auc'] > best_value:
                best_value = all_metrics['prc_auc']

            trial.report(all_metrics['prc_auc'], epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned

            logger.info(f'test metrics: {all_metrics}')

    return best_value


def trial_task():
    study = optuna.create_study(
        storage="sqlite:///multi_task.sqlite3",  # 数据文件存放的地址
        study_name="multi_task",  # 需要指定学习任务的名字，该名字就是数据文件的名字
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        load_if_exists=True)
    study.optimize(my_task, n_trials=12)

    print(f'best params:{study.best_params}, best value:{study.best_value}, best trial:{study.best_trial}')


if __name__ == '__main__':
    trial_task()

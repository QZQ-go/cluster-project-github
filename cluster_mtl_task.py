import copy
import shutil
import pandas
import os
import matplotlib.pyplot as plt
import torch
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
    def __init__(self, limit_head=None):
        super().__init__()

        if not limit_head:
            self.cluster_list = list(range(cluster_num))
        else:
            self.cluster_list = limit_head

        self.body = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(151, 512)),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout(0.3)),

                    ("linear2", nn.Linear(512, 256)),
                    ("relu2", nn.ReLU()),
                    ("dropout2", nn.Dropout(0.3)),

                    ("linear3", nn.Linear(256, 64)),
                    ("relu3", nn.ReLU()),
                    ("dropout3", nn.Dropout(0.3)),
                ]
            )
        )

        self.head_pattern = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(64, 1)),
                    ("Sigmoid", nn.Sigmoid()),
                ]
            )
        )

        self.multi_head_dict = nn.ModuleDict({str(clst): copy.deepcopy(self.head_pattern) for clst in self.cluster_list})

    def forward(self, x):
        """
        :param x: 由于需要进行分头训练，所以数据是按照字典的方式储存的，需要转译一下
        :return:
        """
        head_res = {}
        for clst in self.cluster_list:
            _x = self.body(x[clst])
            _x = self.multi_head_dict[str(clst)](_x)
            head_res.update({clst: _x})
        return head_res


def my_task(prefix='mts_dbscan6',
            limit_head=None):

    # 引入limit_head可以限制多任务的任务头数量，从而实现更多的对比试验
    if not limit_head:
        cluster_list = list(range(cluster_num))
    else:
        cluster_list = limit_head

    BATCH_SIZE = 512
    POS_WEIGHT = 30
    NEG_WEIGHT = 10
    LR = 0.06
    momentum = 0.01
    weight_decay = 1.1e-4
    step_size = 8
    gamma = 0.7
    random_seed = 5

    EPOCHS = 100
    AIMED_ACC = 0.7
    AIMED_REC = 0.7
    NUM_WORK = 0
    PIN_MEMORY = True

    PREFIX = prefix

    t = 'files/'
    if not os.path.exists(t):
        os.mkdir(t)

    target_dir = os.path.join(t, f'{PREFIX}/')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    else:
        shutil.rmtree(target_dir)
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
                     product(cluster_list, ['train', 'val'], ['loss', 'rec', 'acc'])}

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
    model = MultiTaskDnn(limit_head)
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
            dse_t_metrics = {clst: MetricsRecoder() for clst in cluster_list}

            target_data_loader = data_loader_dict[mode]
            for fold_num, data in enumerate(tqdm(target_data_loader)):
                if mode == "train":
                    optimizer.zero_grad()

                label: dict = data[1]
                inputs: dict = data[0]
                for k in cluster_list:
                    inputs[k] = inputs[k].to(device)
                    label[k] = label[k].float().to(device)

                outputs: dict = model(inputs)
                for k in cluster_list:
                    outputs[k] = outputs[k].view(outputs[k].shape[0])

                loss_list = []
                for k in cluster_list:
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
            dse_res_dict = {clst: dse_t_metrics[clst].get_metrics() for clst in cluster_list}

            recorder[f'{mode}_loss'].append(total_loss)
            recorder[f'{mode}_rec'].append(rec)
            recorder[f'{mode}_acc'].append(acc)

            logger.info(f"{mode} loss {total_loss}")
            logger.info(f"{mode} acc {acc}")
            logger.info(f"{mode} rec {rec}")

            for clst in cluster_list:
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

        # 每10次运行一次测试集
        if epoch % 10 == 0 and os.path.exists(MODEL_FILE_NAME):
            logger.info('start testing!')  # 当模型训练结束，加载最优参数进行结果测试
            test_model = MultiTaskDnn(limit_head)
            test_model.load_state_dict(torch.load(MODEL_FILE_NAME))
            test_model.to(device)
            test_model.eval()

            metrics_dict = {str(clst): TestMetricsRecoder() for clst in cluster_list}
            metrics_dict.update({'all': TestMetricsRecoder()})

            for fold_num, data in enumerate(tqdm(data_loader_dict['test'])):
                label: dict = data[1]
                inputs: dict = data[0]
                for k in cluster_list:
                    inputs[k] = inputs[k].to(device)
                    label[k] = label[k].float().to(device)

                outputs: dict = test_model(inputs)

                for k in cluster_list:
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

            logger.info(f'test metrics: {all_metrics}')

    return best_value


def single_head_task():
    for i in range(3, 6):
        limit_head = [i]
        prefix = f'gmm_head_{i}'
        print(f'proc {prefix}')
        my_task(prefix, limit_head)


if __name__ == '__main__':
    single_head_task()

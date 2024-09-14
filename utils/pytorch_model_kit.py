import gc
import numpy
import optuna
import pandas
import matplotlib.pyplot as plt
from torch import torch
from tqdm import tqdm
from sklearn import metrics
from config import device


def smooth_curve(x, y):
    """
    如果测试的数据太多的话，使用sklearn的metrics包计算auc的时候容易报错
    ValueError: x is neither increasing nor decreasing
    所以要针对这个问题处理一下两个序列
    """
    flag = x[0] > x[-1]  # flag是true的话是降序，false为升序
    target_x, target_y = [], []
    for i in range(len(x)):
        if len(target_x) == 0:
            target_x.append(x[i])
            target_y.append(y[i])
        elif flag:
            if x[i] < target_x[-1]:
                target_x.append(x[i])
                target_y.append(y[i])
        else:
            if x[i] > target_x[-1]:
                target_x.append(x[i])
                target_y.append(y[i])
    return target_x, target_y


class MetricsRecoder:
    def __init__(self):
        self.total_loss = []
        self.total_acc_num = 0
        self.total_num = 0
        self.total_rec_ture = 0
        self.total_rec_num = 0

    def load(self, labels, outputs, loss):
        # 计算准确率
        acc_item = outputs.round() == labels
        self.total_acc_num += sum(acc_item).item()
        self.total_num += len(acc_item)

        # 召回率计算
        # 如果被标记为死亡，阳标签个数+1
        pos_num = sum(labels == 1).item()
        # 如果对应的位置模型预测也为1，那么记为召回率正确+1
        rec_num = sum(
            torch.where(labels == 1,
                        outputs.round(),
                        torch.zeros(labels.shape[0]).to(device))
        ).item()

        self.total_rec_num += pos_num
        self.total_rec_ture += rec_num
        self.total_loss.append(loss.item())

    def get_metrics(self):
        rec = round(self.total_rec_ture / self.total_rec_num, 5)
        acc = round(self.total_acc_num / self.total_num, 5)
        total_loss = numpy.mean(self.total_loss)
        return acc, rec, total_loss


class TestMetricsRecoder(MetricsRecoder):
    def __init__(self):
        super().__init__()

        self.y_labels = []
        self.y_outputs = []

    def load(self, labels, outputs, loss):
        # 记录auc数据
        self.y_labels.append(labels)
        self.y_outputs.append(outputs)

    def get_metrics(self, prc_save_path=None):
        if prc_save_path is None:
            prc_save_path = 'files/precision_recall_curve.csv'

        if isinstance(self.y_labels[0], torch.Tensor):
            total_label = torch.cat(self.y_labels).tolist()
            total_score = torch.cat(self.y_outputs).tolist()
        elif isinstance(self.y_labels[0], numpy.ndarray):
            total_label = numpy.concatenate(self.y_labels).tolist()
            total_score = torch.concatenate(self.y_outputs).tolist()
        else:
            total_label = self.y_labels
            total_score = self.y_outputs

        y_pred = numpy.round(total_score, decimals=0)
        acc = metrics.accuracy_score(total_label, y_pred)
        rec = metrics.recall_score(total_label, y_pred)
        pre = metrics.precision_score(total_label, y_pred, average='binary')
        g_means = numpy.sqrt(pre * rec)  # g-means的计算方法就是精确度和召回率相乘后开根号
        mcc = metrics.matthews_corrcoef(total_label, y_pred)
        lr_precision, lr_recall, threshold = metrics.precision_recall_curve(total_label, total_score)
        min_len = numpy.min([len(lr_precision), len(lr_recall), len(threshold)])
        _df = pandas.DataFrame({'lr_precision': lr_precision[:min_len],
                                'lr_recall': lr_recall[:min_len],
                                'threshold': threshold[:min_len]})
        _df.to_csv(prc_save_path)
        lr_pr, lr_rec = smooth_curve(lr_precision, lr_recall)
        prc_auc = metrics.auc(lr_pr, lr_rec)
        result = {'rec': rec, 'acc': acc, 'prc_auc': prc_auc, 'mcc': mcc, 'g_means': g_means}
        return result


class PyTorchTrainer:
    def __init__(self,
                 model, loss_func, optimizer, scheduler,
                 train_data_loader, val_data_loader, test_data_loader,
                 epoch,
                 procedure_csv_path, test_csv_path, image_path, model_path, prc_path,
                 pos_weight, neg_weight,
                 logger,
                 if_trial=False, trial=None):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

        self.epoch = epoch

        self.procedure_csv_path = procedure_csv_path
        self.test_csv_path = test_csv_path
        self.image_path = image_path
        self.model_path = model_path
        self.prc_path = prc_path

        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

        self.logger = logger

        self.if_trial = if_trial
        self.trial = trial

        self.recorder = {}
        for _type in ['train', 'val']:
            self.recorder.update(
                {
                    f'{_type}_loss': [],
                    f'{_type}_rec': [],
                    f'{_type}_acc': []
                })

    def _unit_train(self, model, mode, target_data_loader, target_metrics_class=MetricsRecoder):
        """
        最小轮的训练单位
        """
        model.train() if mode == "train" else model.eval()
        my_metrics = target_metrics_class()

        for i, data in enumerate(tqdm(target_data_loader)):
            if mode == "train":
                self.optimizer.zero_grad()

            labels = data[1].float().to(device)
            inputs = data[0].float().to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.shape[0])

            loss = self.calculate_loss(outputs, labels)
            if mode == "train":
                loss.backward()
                self.optimizer.step()

            my_metrics.load(labels.detach(), outputs.detach(), loss.detach())
        return my_metrics

    def train(self):
        if self.if_trial:
            best_value = 0
        for epoch in range(self.epoch):
            self.logger.info(f"epoch {epoch}")
            torch.cuda.empty_cache()  # 释放显存
            for mode in ["train", "val"]:
                target_data_loader = self.train_data_loader if mode == "train" else self.val_data_loader
                my_metrics = self._unit_train(self.model, mode, target_data_loader)
                acc, rec, total_loss = my_metrics.get_metrics()

                self.logger.info(f"{mode} loss {total_loss}")
                self.logger.info(f"{mode} acc {acc}")
                self.logger.info(f"{mode} rec {rec}")
                self.recorder[f'{mode}_loss'].append(total_loss)
                self.recorder[f'{mode}_rec'].append(rec)
                self.recorder[f'{mode}_acc'].append(acc)

            self.scheduler.step()

            self.save_recorder_image()
            self.save_recorder_csv()
            self.save_model_params()

            if self.if_trial and epoch % 5 == 0:
                test_val = self.model_test(self.model)
                self.logger.info(f'start trail with epoch {epoch} and value {test_val}')
                self.trial.report(test_val, epoch)
                if best_value < test_val:
                    best_value = test_val
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned

        if self.if_trial:
            return best_value
        else:
            return self.model_test(self.model)

    def save_recorder_image(self):
        """
        记录该时刻训练器产生的结果并保存为图像文件
        """
        fig = plt.figure()
        ax_loss, ax_acc, ax_rec = fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(224)

        ax_loss.set(title='Loss')
        ax_acc.set(title='Acc')
        ax_rec.set(title='Rec')

        for _type in ["train", "val"]:
            ax_loss.plot(self.recorder[f'{_type}_loss'], label=_type)
            ax_acc.plot(self.recorder[f'{_type}_acc'], label=_type)
            ax_rec.plot(self.recorder[f'{_type}_rec'], label=_type)

            plt.legend()
            plt.savefig(self.image_path)

    def save_recorder_csv(self):
        """
        记录该时刻训练器产生的结果并保存为csv
        """
        df = pandas.DataFrame(self.recorder)
        df.to_csv(self.procedure_csv_path)

    def save_model_params(self):
        torch.save(self.model.state_dict(), self.model_path)

    def calculate_loss(self, outputs, labels):
        weight_tensor = torch.where(
            (labels == 1) & (outputs.round() == 0), self.pos_weight, 1
        )
        weight_tensor = torch.where(
            (labels == 1) & (outputs.round() == 1), self.neg_weight, weight_tensor
        )

        self.loss_func.weight = weight_tensor.to(device)
        loss = self.loss_func(outputs, labels)
        return loss

    def model_test(self, model) -> float:
        my_metrics = self._unit_train(model,
                                      mode='test',
                                      target_data_loader=self.test_data_loader,
                                      target_metrics_class=TestMetricsRecoder)
        metrics_result = my_metrics.get_metrics(self.prc_path)
        print(metrics_result)
        _df = pandas.DataFrame([metrics_result])
        _df.to_csv(self.test_csv_path)
        return metrics_result['mcc']


class PyTorchTrainerSVM(PyTorchTrainer):
    def __init__(self, model, loss_func, optimizer, scheduler, train_data_loader, val_data_loader, test_data_loader,
                 epoch, procedure_csv_path, test_csv_path, image_path, model_path, pos_weight, neg_weight, logger):
        super().__init__(model, loss_func, optimizer, scheduler, train_data_loader, val_data_loader, test_data_loader,
                         epoch, procedure_csv_path, test_csv_path, image_path, model_path, pos_weight, neg_weight,
                         logger)

    def calculate_loss(self, outputs, labels):
        weight_tensor = torch.where(
            (labels == 1) & (outputs.round() == 0), self.pos_weight, 1
        )
        weight_tensor = torch.where(
            (labels == 1) & (outputs.round() == 1), self.neg_weight, weight_tensor
        )

        # 第一部分的loss使距离求得最小
        L2 = torch.matmul(self.model.linear.weight, self.model.linear.weight.T)
        # 第二部分的loss求误分类的损失
        classification_term = torch.mean(
            weight_tensor * torch.maximum(torch.tensor((0.)), 1. - outputs * labels))

        loss = L2 + classification_term
        return loss


class PyTorchTrainerNoWeight(PyTorchTrainer):
    def __init__(self, model, loss_func, optimizer, scheduler, train_data_loader, val_data_loader, test_data_loader,
                 epoch, procedure_csv_path, test_csv_path, image_path, model_path, pos_weight, neg_weight, logger):
        super().__init__(model, loss_func, optimizer, scheduler, train_data_loader, val_data_loader, test_data_loader,
                         epoch, procedure_csv_path, test_csv_path, image_path, model_path, pos_weight, neg_weight,
                         logger)

    def calculate_loss(self, outputs, labels):
        return self.loss_func(outputs, labels)


if __name__ == '__main__':
    pass

import logging
import os
import sys
from datetime import datetime

import pandas as pd
import torch
import time
import numpy as np

from trainer.utils import masked_mae_torch, ensure_dir, masked_rmse_torch, masked_mape_torch


class Trainer:
    def __init__(self, config, model, data_feature):
        self.config = config
        self._scaler = data_feature.get('scaler')
        self.steps_per_epoch = data_feature.get("num_batches")
        self.device = self.config.get('device', torch.device('cpu'))
        self.pct_start = config.get("pct_start", 0.2)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.weight_decay = self.config.get('weight_decay', 0)
        self.lr_beta1 = self.config.get('lr_beta1', 0.9)
        self.lr_beta2 = self.config.get('lr_beta2', 0.999)
        self.lr_betas = (self.lr_beta1, self.lr_beta2)
        self.lr_epsilon = self.config.get('lr_epsilon', 1e-8)
        self.epochs = self.config.get('max_epoch', 100)
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.patience = self.config.get('patience', 50)
        self.clip_grad_norm = self.config.get('clip_grad_norm', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self.lr_decay = self.config.get('lr_decay', False)
        self._logger = self.get_logger(config)
        self.save_pred = self.config.get('save_pred', False)
        self.cache_dir = "./cache/" + config['exp_id'] + "/model"
        self.evaluate_res_dir = "./cache/" + config['exp_id'] + "/evaluate"
        self.metrics = ["MAE", "RMSE", "MAPE"]

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                                                steps_per_epoch=self.steps_per_epoch,
                                                                pct_start=self.pct_start,
                                                                epochs=self.epochs,
                                                                max_lr=self.learning_rate)

    def get_logger(self, config, name=None):
        """
        获取Logger对象

        Args:
            config(ConfigParser): config
            name: specified name

        Returns:
            Logger: logger
        """
        log_dir = './log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_filename = '{}.log'.format(config['exp_id'])
        logfilepath = os.path.join(log_dir, log_filename)

        logger = logging.getLogger(name)

        log_level = config.get('log_level', 'INFO')

        if log_level.lower() == 'info':
            level = logging.INFO
        elif log_level.lower() == 'debug':
            level = logging.DEBUG
        elif log_level.lower() == 'error':
            level = logging.ERROR
        elif log_level.lower() == 'warning':
            level = logging.WARNING
        elif log_level.lower() == 'critical':
            level = logging.CRITICAL
        else:
            level = logging.INFO

        logger.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(logfilepath)
        file_handler.setFormatter(formatter)

        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info('Log directory: %s', log_dir)
        return logger

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        self._logger.info("Loaded model at " + cache_name)
        model_state, optimizer_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def save_model_with_epoch(self, epoch):
        """
        保存某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        ensure_dir(self.cache_dir)
        config = dict()
        config['model_state_dict'] = self.model.state_dict()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        config['epoch'] = epoch
        model_path = self.cache_dir + '/' + 'epoch%d.tar' % epoch
        torch.save(config, model_path)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path

    def load_model_with_epoch(self, epoch):
        """
        加载某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        model_path = self.cache_dir + '/' + 'epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch))

    def train(self, train_dataloader, eval_dataloader):
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        for epoch_idx in range(self.epochs):
            start_time = time.time()
            losses = self._train_epoch(train_dataloader, epoch_idx)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx)
            end_time = time.time()
            eval_time.append(end_time - t2)


            log_lr = self.optimizer.param_groups[0]['lr']
            message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                format(epoch_idx, self.epochs, np.mean(losses), val_loss, log_lr, (end_time - start_time))
            self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                model_file_name = self.save_model_with_epoch(epoch_idx)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                  'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))

        self.load_model_with_epoch(best_epoch)
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx):
        """
        完成模型一个轮次的训练

        Args:
            train_dataloader: 训练数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            list: 每个batch的损失的数组
        """

        def adjust_learning_rate(optimizer, scheduler, epoch):
            lr_adjust = {epoch: scheduler.get_last_lr()[0]}
            if epoch in lr_adjust.keys():
                lr = lr_adjust[epoch]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        self.model.train()
        loss_func = masked_mae_torch
        losses = []
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            output = self.model.predict(batch)
            y_true = self._scaler.inverse_transform(batch['y'])
            y_pred = self._scaler.inverse_transform(output)
            loss = loss_func(y_true, y_pred, null_val=0.0)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.lr_decay:
                adjust_learning_rate(self.optimizer, self.lr_scheduler, epoch_idx + 1)
                self.lr_scheduler.step()
        return losses

    def _valid_epoch(self, eval_dataloader, epoch_idx):
        """
        完成模型一个轮次的评估

        Args:
            eval_dataloader: 评估数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            float: 评估数据的平均损失值
        """
        with torch.no_grad():
            self.model.eval()
            loss_func = masked_mae_torch
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                output = self.model.predict(batch)
                y_true = self._scaler.inverse_transform(batch['y'])
                y_pred = self._scaler.inverse_transform(output)
                loss = loss_func(y_true, y_pred, null_val=0.0)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            return mean_loss

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model.eval()
            # self.evaluator.clear()
            y_truths = []
            y_preds = []
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                output = self.model.predict(batch)
                y_true = self._scaler.inverse_transform(batch['y'])
                y_pred = self._scaler.inverse_transform(output)
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())
                # evaluate_input = {'y_true': y_true, 'y_pred': y_pred}
                # self.evaluator.collect(evaluate_input)
            # self.evaluator.save_result(self.evaluate_res_dir)
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)  # concatenate on batch
            outputs = {'prediction': y_preds, 'truth': y_truths}
            if self.save_pred:
                filename = \
                    time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                    + self.config["exp_id"] + '_predictions.npz'
                np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **outputs)
            self.get_metric({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})
            test_result = self.save_result(self.evaluate_res_dir)
            return test_result

    def get_metric(self, batch):
        y_true = batch['y_true']  # tensor
        y_pred = batch['y_pred']  # tensor
        self.result = {}
        self.intermediate_result = {}
        self.timeslots =y_true.shape[1]
        for i in range(1, self.timeslots + 1):
            for metric in self.metrics:
                if metric == 'MAE':
                    self.intermediate_result[metric + '@' + str(i)].append(
                        masked_mae_torch(y_pred[:, :i], y_true[:, :i], 0).item())
                elif metric == 'RMSE':
                    self.intermediate_result[metric + '@' + str(i)].append(
                        masked_rmse_torch(y_pred[:, :i], y_true[:, :i], 0).item())
                elif metric == 'MAPE':
                    self.intermediate_result[metric + '@' + str(i)].append(
                        masked_mape_torch(y_pred[:, :i], y_true[:, :i], 0).item())
        for i in range(1, self.timeslots + 1):
            for metric in self.metrics:
                self.result[metric + '@' + str(i)] = sum(self.intermediate_result[metric + '@' + str(i)]) / \
                                                     len(self.intermediate_result[metric + '@' + str(i)])
        return self.result

    def save_result(self, save_path, filename=None):
        """
        将评估结果保存到 save_path 文件夹下的 filename 文件中

        Args:
            save_path: 保存路径
            filename: 保存文件名
        """
        ensure_dir(save_path)
        if filename is None:  # 使用时间戳
            filename = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config['model'] + '_' + self.config['dataset']

        dataframe = {}
        for metric in self.metrics:
            dataframe[metric] = []
        for i in range(1, self.timeslots + 1):
            for metric in self.metrics:
                dataframe[metric].append(self.result[metric + '@' + str(i)])
        dataframe = pd.DataFrame(dataframe, index=range(1, self.timeslots + 1))
        dataframe.to_csv(os.path.join(save_path, '{}.csv'.format(filename)), index=False)
        self._logger.info('Evaluate result is saved at ' +
                          os.path.join(save_path, '{}.csv'.format(filename)))
        self._logger.info("\n" + str(dataframe))
        return dataframe
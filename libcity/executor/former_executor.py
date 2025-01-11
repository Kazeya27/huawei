import os
import time
from pdb import set_trace

import numpy as np
import torch

from libcity.executor import TrafficStateExecutor


class FormerExecutor(TrafficStateExecutor):

    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)

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
            attns = []
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                attn, output = self.model(batch)
                y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
                y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())
                attns.append(attn.cpu().numpy())
                # evaluate_input = {'y_true': y_true, 'y_pred': y_pred}
                # self.evaluator.collect(evaluate_input)
            # self.evaluator.save_result(self.evaluate_res_dir)
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)  # concatenate on batch
            attns = np.concatenate(attns, axis=0)
            # attns取平均
            attns = np.mean(attns, axis=0)
            # 保存attns
            set_trace()
            np.save(self.evaluate_res_dir + '/attns.npy', attns)
            self.evaluator.clear()
            self.evaluator.collect({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})
            test_result = self.evaluator.save_result(self.evaluate_res_dir)
            return test_result

    def _valid_epoch(self, eval_dataloader, epoch_idx, loss_func=None):
        with torch.no_grad():
            self.model.eval()
            if loss_func is None:
                loss_func = self.model.calculate_loss
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                loss = loss_func(batch)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss)
            return mean_loss
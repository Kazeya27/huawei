import torch
import numpy as np

from libcity.executor import TrafficStateExecutor


class BIGSTExecutor(TrafficStateExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)

    def _valid_epoch(self, eval_dataloader, epoch_idx, loss_func=None):
        with torch.no_grad():
            self.model.eval()
            loss_func = loss_func if loss_func is not None else self.model.calculate_loss_without_predict
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                output = self.model.predict(batch)
                loss = loss_func(batch["y"], output)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss, epoch_idx)
            return mean_loss
"""
Example model that is managed by ModelManager
"""
from libs.models.Model import Model
from config.constants import HyperParamKey, LoaderParamKey
from config.basic_conf import DEVICE
from libs.models.modules import BoW
import torch.nn.functional as F
import logging

logger = logging.getLogger('__main__')


class BagOfWords(Model):
    def __init__(self, hparams, lparams, io_paths, alias='scratch'):
        super().__init__(hparams, lparams, io_paths, alias)
        self.model = BoW.BagOfWords(lparams[LoaderParamKey.ACT_VOCAB_SIZE]
                                    , hparams[HyperParamKey.EMBEDDING_DIM]).to(DEVICE)

    def eval_model(self, dataloader):
        if self.model is None:
            raise AssertionError("cannot evaluate model: %s, it was never initialized" % self.name)
        else:
            correct = 0
            total = 0
            cur_loss = 0
            self.model.eval()  # good practice to set the model to evaluation mode (no dropout)
            for data, lengths, labels in dataloader:
                data_batch, length_batch, label_batch = data, lengths, labels
                outputs = F.softmax(self.model(data_batch, length_batch), dim=1)
                predicted = outputs.max(1, keepdim=True)[1]
                cur_loss += F.cross_entropy(outputs, labels).cpu().detach().numpy()

                total += labels.size(0)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
            return 100 * correct / total, cur_loss

    def check_early_stop(self):
        val_acc_history = self.train_curves[self.VAL_ACC]
        t = self.hparams[HyperParamKey.EARLY_STOP_LOOK_BACK]
        required_progress = self.hparams[HyperParamKey.EARLY_STOP_REQ_PROG]

        if len(val_acc_history) >= t + 1 and val_acc_history[-t - 1] > max(val_acc_history[-t:]) - required_progress:
            return True
        return False

    def save(self, io_paths):
        pass
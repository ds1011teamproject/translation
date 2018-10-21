"""
Example model that is managed by ModelManager
"""
from libs.models.BaseModel import BaseModel
from config.constants import HyperParamKey, LoaderParamKey
from config.basic_conf import DEVICE
from libs.models.modules import BoW
import torch.nn.functional as F
import logging

logger = logging.getLogger('__main__')


class BagOfWords(BaseModel):
    """
    example implementation of a model handler
    can optionally also overload the train method if you want to use your own
    """
    def __init__(self, hparams, lparams, cparams, label='scratch', nolog=False):
        super().__init__(hparams, lparams, cparams, label, nolog)
        self.model = BoW.BagOfWords(lparams[LoaderParamKey.ACT_VOCAB_SIZE]
                                    , hparams[HyperParamKey.EMBEDDING_DIM]).to(DEVICE)

    def eval_model(self, dataloader):
        """
        takes all of the data in the loader and forward passes through the model
        :param dataloader: the torch.utils.data.DataLoader with the data to be evaluated
        :return: tuple of (accuracy, loss)
        """
        if self.model is None:
            raise AssertionError("cannot evaluate model: %s, it was never initialized" % self.label)
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
        """
        the method called by the standard training loop in BaseModel to determine early stop
        if no early stop is wanted, can just return False
        can also use the hparam to control whether early stop is considered
        :return: bool whether to stop the loop
        """
        val_acc_history = self.iter_curves[self.VAL_ACC]
        t = self.hparams[HyperParamKey.EARLY_STOP_LOOK_BACK]
        required_progress = self.hparams[HyperParamKey.EARLY_STOP_REQ_PROG]

        if len(val_acc_history) >= t + 1 and val_acc_history[-t - 1] > max(val_acc_history[-t:]) - required_progress:
            return True
        return False


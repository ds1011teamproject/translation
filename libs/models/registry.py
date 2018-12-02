"""
registry for model constructors, used by ModelManager to lookup model constructors
"""
from libs.models.CNN import CNN
from libs.models.CNN2 import CNN_Pool, CNNAttn_Pool, CNN_Tanh, CNNAttn_Tanh, CNN_Relu, CNNAttn_Relu
from libs.models.RNN_GRU import RNN_GRU
from libs.models.RNN_Attention import RNN_Attention
# todo make the registry dynamic based on the file name

reg = {
    'RNN_GRU': RNN_GRU,
    'RNN_Attention': RNN_Attention,
    'CNNAttn_Tanh': CNNAttn_Tanh
}


class ModelRegister:
    def __init__(self):
        for k in reg.keys():
            setattr(self, k, k)

    @property
    def model_list(self):
        return '\n=== Models Available ===\n{}\n========================'.format(
            '\n'.join(self.__dict__.keys()))


modelRegister = ModelRegister()

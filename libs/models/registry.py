"""
registry for model constructors, used by ModelManager to lookup model constructors
"""
from libs.models.RNN_GRU import RNN_GRU
# todo make the registry dynamic based on the file name

reg = {
    'RNN_GRU': RNN_GRU
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

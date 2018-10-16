"""
registry for model constructors, used by ModelManager to lookup model constructors
"""
from libs.models.GRU import GRU

# todo make the registry dynamic based on the file name

reg = {
    'GRU': GRU
}

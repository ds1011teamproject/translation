"""
Entry point for the program, argparse
"""

from lib import ModelManager as mm
# todo: implememnt argparse
import argparse
import logging
from config import basic_conf as conf
conf.init_logger()
logger = logging.getLogger('__main__')


# todo --- MAIN HERE ---
mgr = mm.ModelManager()
mgr.load_data()
mgr.set_model('GRU')
mgr.train()





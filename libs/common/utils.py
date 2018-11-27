import logging
import logging.config

from config.constants import LogConfig

LOG_LEVEL_DEFAULT = getattr(logging, LogConfig['handlers']['default']['level'])


def init_logger(loglevel=LOG_LEVEL_DEFAULT, logfile=None):
    logging.getLogger('__main__').setLevel(loglevel)
    if logfile is None:
        LogConfig['loggers']['']['handlers'] = ['console']
        LogConfig['handlers']['default']['filename'] = 'mt.log'
    else:
        LogConfig['loggers']['']['handlers'] = ['console', 'default']
        LogConfig['handlers']['default']['filename'] = logfile
    logging.config.dictConfig(LogConfig)

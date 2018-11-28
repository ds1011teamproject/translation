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
    LogConfig['handlers']['default']['level'] = loglevel
    logging.config.dictConfig(LogConfig)


def hparam_to_label(prefix, hparam_dict):
    for k in hparam_dict:
        prefix += '-{}{}'.format(k[:5], hparam_dict[k])
    return prefix

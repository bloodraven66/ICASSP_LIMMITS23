import logging
import os
import coloredlogs
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['COLOREDLOGS_LOG_FORMAT']='[%(asctime)s] %(message)s'
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')
coloredlogs.install(level='INFO', logger=logger)

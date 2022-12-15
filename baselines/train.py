"""

Authors
 * Sathvik Udupa 2022 (sathvikudupa66@gmail.com)

"""
  

import os, sys
from utils import common

def main():
    cfg = common.load_config(configfile)
    loaders, trainer, model, cfg = common.load(cfg)
    trainer.main(cfg, model, loaders)

if __name__ == '__main__':
    configfile = sys.argv[1]
    main()  
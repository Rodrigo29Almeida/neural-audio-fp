# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" runDEBUG.py """
import os
import sys
import pathlib
import click
import yaml
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def load_config(config_fname):
    config_filepath = './config/' + config_fname + '.yaml'
    if os.path.exists(config_filepath):
        print(f'cli: Configuration from {config_filepath}')
    else:
        sys.exit(f'cli: ERROR! Configuration file {config_filepath} is missing!!')

    with open(config_filepath, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def update_config(cfg, key1: str, key2: str, val):
    cfg[key1][key2] = val
    return cfg


def print_config(cfg):
    os.system("")
    print('\033[36m' + yaml.dump(cfg, indent=4, width=120, sort_keys=False) +
          '\033[0m')
    return



def train(checkpoint_name, config, max_epoch):
    """ Train a neural audio fingerprinter.

    ex) python run.py train CHECKPOINT_NAME --max_epoch=100

        # with custom config file
        python run.py train CHECKPOINT_NAME --max_epoch=100 -c CONFIG_NAME

    NOTE: If './LOG_ROOT_DIR/checkpoint/CHECKPOINT_NAME already exists, the training will resume from the latest checkpoint in the directory.

    """
    from model.utils.config_gpu_memory_lim import allow_gpu_memory_growth
    from model.trainer import trainer

    cfg = load_config(config)
    if max_epoch: update_config(cfg, 'TRAIN', 'MAX_EPOCH', max_epoch)
    print_config(cfg)
    #allow_gpu_memory_growth()
    trainer(cfg, checkpoint_name)



checkpoint_name = 'CHECKPOINTDEBUG'
config = 'default'
max_epoch = 5
train(checkpoint_name, config, max_epoch)
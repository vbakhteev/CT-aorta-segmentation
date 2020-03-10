import os
import yaml
import argparse
import pytorch_lightning as pl
from easydict import EasyDict


def load_config(config_path):
    with open(config_path, 'r') as fid:
        config = yaml.load(fid, Loader=yaml.FullLoader)
    config = EasyDict(config)
    return config


def parse_args():
    description = 'Segmentation pipeline'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', dest='config_file', type=str, required=True,
                        help='Path of config file',)
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

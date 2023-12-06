import os.path as osp

from Load_reddit import Reddit
#Reddit的读取样例
dataset_name = 'Reddit'
DATA_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
path = osp.join(DATA_DIR , dataset_name)
dataset = Reddit(path)

data = dataset[0]


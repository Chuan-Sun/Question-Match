import os
import argparse
import numpy as np
import pandas as pd
import pathlib
# import networkx as nx
import Levenshtein
from tqdm import tqdm
from collections import deque
from typing import List, Tuple
# import pycorrector
# from pycorrector.macbert.macbert_corrector import MacBertCorrector

# from zhconv import langconv

parser = argparse.ArgumentParser()
parser.add_argument("--train_set",            type=str, required=True, help="The full path of train_set_file")
parser.add_argument("--pseudo_set",           type=str, help="The full path of train_set_file")
parser.add_argument("--dev_set",              type=str, required=True, help="The full path of dev_set_file")
parser.add_argument("--test_set",             type=str, required=True, help="The full path of test_set_file")
parser.add_argument("--save_dir",             default='../data/', type=str, help="The output directory where the model checkpoints will be written.")

args = parser.parse_args()


class DataLoader:
    def __init__(self):
        self.input_col = {
            'train':  ['query1', 'query2', 'label'],
            'pseudo': ['query1', 'query2', 'logit0', 'logit1', 'prob0', 'prob1', 'label'],
            'dev':    ['query1', 'query2', 'label'],
            'test':   ['query1', 'query2'],
        }
        self.output_col  = {
            'train':  ['query1', 'query2', 'logit0', 'logit1', 'prob0', 'prob1', 'label', 'pseudo', 'weight'],
            'pseudo': ['query1', 'query2', 'logit0', 'logit1', 'prob0', 'prob1', 'label', 'pseudo', 'weight'],
            'dev':    ['query1', 'query2', 'label'],
            'test':   ['query1', 'query2'],
        }

    def load(self, path, mode):
        input_col  = self.input_col[mode]
        output_col = self.output_col[mode]

        data = pd.read_table(path, names=input_col, na_filter=False, quoting=3)

        if 'pseudo' in output_col:
            if mode == 'pseudo':
                data['pseudo'] = 1
            else:
                data['pseudo'] = 0

        if 'weight' in output_col:
            data['weight'] = 1.0

        new_col = set(output_col) - set(input_col)

        if 'logit0' in new_col:
            data['logit0'] = 0.0
            data['logit1'] = 0.0

        if 'prob0' in new_col:
            mask = data['label'] == 0
            data['prob0'] = mask * 0.95 + ~mask * 0.05
            data['prob1'] = mask * 0.05 + ~mask * 0.95

        data = data[output_col]
        assert data.notnull().all().all()
        return data


def preprocess(input_path, output_path, mode):
    if isinstance(input_path, list):
        assert isinstance(mode, list) and len(input_path) == len(mode)
        data = []
        for p, m in zip(input_path, mode):
            d = DataLoader().load(p, m)
            data.append(d)
        data = pd.concat(data, ignore_index=True)
    else:
        data = DataLoader().load(input_path, mode)

    # 输出
    data.to_csv(output_path, sep='\t', header=False, index=False, quoting=3, line_terminator='\n')


def main():
    save_dir = pathlib.Path(args.save_dir)

    print('\n正在处理训练集数据')
    if args.pseudo_set:
        train_paths = [args.train_set, args.pseudo_set]
        train_modes = ['train', 'pseudo']
    else:
        train_paths = args.train_set
        train_modes = 'train'
    preprocess(train_paths,
               save_dir / 'train.txt',
               train_modes)

    print('\n正在处理验证集数据')
    preprocess(args.dev_set,   save_dir / 'dev.txt',   'dev')

    print('\n正在处理测试集数据')
    preprocess(args.test_set,  save_dir / 'test.txt',  'test')


if __name__ == '__main__':
    main()
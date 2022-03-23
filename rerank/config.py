import os
import sys

join = os.path.join
dirname = os.path.dirname

class BASIC_ARGS:
    RANDOM_STATE = 0

class PATH_ARGS:
    # path
    ROOT_DIR = os.path.dirname(__file__)
    DATA_DIR = join(ROOT_DIR, 'dataset')
    RAW_DATA_DIR = join(ROOT_DIR, 'data')
    MODEL_DIR = join(ROOT_DIR, 'model')
    MODEL_FILE_DIR = join(ROOT_DIR, 'model_file')


class Bert_Args:
    # baseline: bert model args
    device = 'cuda'
    model_name = 'hfl/chinese-macbert-base'
    tokenizer = 'hfl/chinese-macbert-base'
    class_nums = 2
    num_workers = 4
    dense_size = 128

class Simbert_Args:
    dict_path = join(PATH_ARGS.MODEL_FILE_DIR, 'chinese_simbert_L-12_H-768_A-12', 'vocab.txt')
    config_path = join(PATH_ARGS.MODEL_FILE_DIR, 'chinese_simbert_L-12_H-768_A-12', 'bert_config.json')
    checkpoint_path = join(PATH_ARGS.MODEL_FILE_DIR, 'chinese_simbert_L-12_H-768_A-12', 'bert_model.ckpt')
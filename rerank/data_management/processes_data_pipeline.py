import os
import sys
from unicodedata import name
join = os.path.join
dirname = os.path.dirname
sys.path.append(join(dirname(__file__), '..'))
from config import *

import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from data_management.eda_data import get_eda_data


class Data_Process_Pipeline:
    def __init__(self) -> None:
        pass

    def export_test_data(self, data_path):
        test_data = pd.read_csv(data_path, sep='\t')
        test_data = shuffle(test_data, random_state=BASIC_ARGS.RANDOM_STATE).iloc[:2000]
        return test_data
    
    def export_eda_test_data(self, data):
        sent_A_eda = get_eda_data(data.A.tolist())
        sent_B_eda = get_eda_data(data.B.tolist())
        labels = data.label.tolist()
        eda_dataset = []
        pbar = tqdm(total=len(labels), desc='merge eda dataset')
        for sent_set_a, sent_set_b, label in zip(sent_A_eda, sent_B_eda, labels):
            for eda_a, eda_b in zip(sent_set_a, sent_set_b):
                eda_dataset.append([eda_a, eda_b, label])
            pbar.update(1)
        eda_dataset = pd.DataFrame(eda_dataset, columns=['A', 'B', 'label'])
        eda_dataset = shuffle(eda_dataset, random_state=BASIC_ARGS.RANDOM_STATE).iloc[:2000]
        return eda_dataset


if __name__ == '__main__':
    dpp = Data_Process_Pipeline()
    # test_data = dpp.read_test_data(join(PATH_ARGS.RAW_DATA_DIR, 'lcqmc/dev.tsv'))
    # test_data.to_csv(join(PATH_ARGS.DATA_DIR, 'baseline_dataset/dev.tsv'), index=False, encoding='gbk', sep='\t')
    
    test_data = pd.read_csv(join(PATH_ARGS.DATA_DIR, 'baseline_dataset/dev.tsv'), encoding='gbk', sep='\t', names=['A', 'B', 'label'])
    eda_dataset = dpp.export_eda_test_data(test_data)
    eda_dataset.to_csv(join(PATH_ARGS.DATA_DIR, 'baseline_dataset/eda_dev.tsv'), index=False, encoding='gbk', sep='\t')

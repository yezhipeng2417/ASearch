import os
import sys
join = os.path.join
dirname = os.path.dirname
sys.path.append(join(dirname(__file__), '..'))
from config import *

import pandas as pd
from sklearn.utils import shuffle


class Data_Process_Pipeline:
    def __init__(self) -> None:
        pass

    def read_test_data(self, data_path):
        test_data = pd.read_csv(data_path, sep='\t')
        test_data = shuffle(test_data, random_state=BASIC_ARGS.RANDOM_STATE).iloc[:2000]
        return test_data


if __name__ == '__main__':
    dpp = Data_Process_Pipeline()
    test_data = dpp.read_test_data(join(PATH_ARGS.RAW_DATA_DIR, 'lcqmc/dev.tsv'))
    test_data.to_csv(join(PATH_ARGS.DATA_DIR, 'baseline_dataset/dev.tsv'), index=False, encoding='gbk', sep='\t')

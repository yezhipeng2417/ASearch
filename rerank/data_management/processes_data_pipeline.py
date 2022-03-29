import os
import sys
join = os.path.join
dirname = os.path.dirname
sys.path.append(join(dirname(__file__), '..'))
from config import *

import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from feature_extract.feature_extract_batch import extract_feature, extract_feature_single
from data_management.eda_data import get_eda_data


class Data_Process_Pipeline:
    def __init__(self) -> None:
        pass

    def export_test_data(self, data_path):
        test_data = pd.read_csv(data_path, sep='\t', names=['A', 'B', 'label'])
        test_data = test_data[['A', 'B']][test_data['label']==1]
        test_data = shuffle(test_data, random_state=BASIC_ARGS.RANDOM_STATE)
        other_data = test_data.iloc[2000:10000].B.tolist()
        test_data = test_data.iloc[:2000]
        return test_data, other_data
    
    def export_eda_test_data(self, data):
        sent_A_eda = get_eda_data(data.A.tolist())
        sent_B_eda = get_eda_data(data.B.tolist())
        eda_dataset = []
        pbar = tqdm(total=len(sent_B_eda), desc='merge eda dataset')
        for sent_set_a, sent_set_b in zip(sent_A_eda, sent_B_eda):
            for eda_a, eda_b in zip(sent_set_a, sent_set_b):
                eda_dataset.append([eda_a, eda_b])
            pbar.update(1)
        eda_dataset = pd.DataFrame(eda_dataset, columns=['A', 'B'])
        eda_dataset = shuffle(eda_dataset, random_state=BASIC_ARGS.RANDOM_STATE).iloc[:2000]
        return eda_dataset
    
    def export_features(self, data_path):
        data = pd.read_csv(data_path, sep='\t', names=['A', 'B', "label"])
        data = shuffle(data, random_state=BASIC_ARGS.RANDOM_STATE)
        data = data.iloc[:10000]
        features = extract_feature_single(data.A.tolist(), data.B.tolist())
        # features = [features[i][i] for i in range(len(features)) if i < len(features[i])]
        df_features = pd.DataFrame(features, columns=['a', 'b', 'c'])
        df_features['label'] = data.label.tolist()[:df_features.shape[0]]
        df_features.to_csv('/mnt/disk/hdd1/similarity/ASearch/rerank/dataset/ctr_cqr_dataset_10000.csv', index=False)
        return features, data.label.tolist()


if __name__ == '__main__':
    dpp = Data_Process_Pipeline()
    test_data, other_data = dpp.export_test_data(join(PATH_ARGS.RAW_DATA_DIR, 'lcqmc/dev.tsv'))
    test_data.to_csv(join(PATH_ARGS.DATA_DIR, 'baseline_dataset/dev.tsv'), index=False, encoding='gbk', sep='\t')
    open(join(PATH_ARGS.DATA_DIR, 'baseline_dataset/dev_other_data.tsv'), 'w').write('\n'.join(other_data))
    
    # test_data = pd.read_csv(join(PATH_ARGS.DATA_DIR, 'baseline_dataset/dev.tsv'), encoding='gbk', sep='\t', names=['A', 'B', 'label'])
    # eda_dataset = dpp.export_eda_test_data(test_data)
    # eda_dataset.to_csv(join(PATH_ARGS.DATA_DIR, 'baseline_dataset/eda_dev.tsv'), index=False, encoding='gbk', sep='\t')

import os, sys
import json
from textwrap import indent

from sklearn import ensemble
join = os.path.join
dirname = os.path.dirname
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import pandas as pd
from config import *
import pickle as pkl
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from feature_extract.feature_extract_batch import extract_feature
from data_management.processes_data_pipeline import Data_Process_Pipeline


def train():
    train_data_path = join(PATH_ARGS.RAW_DATA_DIR, 'lcqmc', 'train.tsv')
    dpp = Data_Process_Pipeline()
    X, y = dpp.export_features(train_data_path)
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=BASIC_ARGS.RANDOM_STATE)

    model = XGBClassifier()
    model.fit(x_train, y_train, verbose=True)
    pkl.dump(model, open(join(PATH_ARGS.MODEL_FILE_DIR, 'simbert_cqr_ctr.pkl'), "wb"))
    model = pkl.load(open(join(PATH_ARGS.MODEL_FILE_DIR, 'simbert_cqr_ctr.pkl'), "rb"))
    y_pred = model.predict(x_test)
    f1_report = classification_report(y_test, y_pred, target_names=['0', '1'])
    print(f1_report)

def evaluate(test_data_path):
    data = pd.read_csv(test_data_path, names=['A', 'B'], encoding='gbk', sep='\t').iloc[:2000]
    data_a = [i for i in data.A.tolist() if i==i]
    data_b = [i for i in data.B.tolist() if i==i]
    features = extract_feature(data_a, data_b)
    feature_length = len(features)
    features = sum(features, [])
    model = pkl.load(open(join(PATH_ARGS.MODEL_FILE_DIR, 'simbert_cqr_ctr.pkl'), "rb"))
    score = [i[1] for i in model.predict_proba(features)]
    score = np.array(score).reshape(feature_length, -1)
    score = [sorted([[ix, item] for ix, item in enumerate(row)], key=lambda x: x[-1], reverse=True) for row in score]
    total_cnt = len(score)
    top_1_cnt = 0
    top_3_cnt = 0
    top_5_cnt = 0
    top_10_cnt = 0
    bad_cases = []
    result = []
    for ix, row in enumerate(score):
        temp_pred_ix = [i[0] for i in row[:10]]
        bad_cases.append([ix, row[:10]])
        if ix in temp_pred_ix[:1]:
            top_1_cnt += 1
            top_3_cnt += 1
            top_5_cnt += 1
            top_10_cnt += 1
        elif ix in temp_pred_ix[:3]:
            top_3_cnt += 1
            top_5_cnt += 1
            top_10_cnt += 1
        elif ix in temp_pred_ix[:5]:
            top_5_cnt += 1
            top_10_cnt += 1
        elif ix in temp_pred_ix[:10]:
            bad_cases.append([ix, row[:10]])
            top_10_cnt += 1
        else:
            bad_cases.append([ix, row[:10]])
            continue
    bad_cases = [[i[0], data_a[i[0]], [[data_b[j[0]], j[1]] for j in i[-1]]] for i in bad_cases]
    bad_cases_json = {}
    for item in bad_cases:
        ix, query, answer = item
        bad_cases_json[ix] = {
            'query': query,
            'answer': [{'text': a[0], 'score': float(a[1])} for a in answer]
        }

    result_json = {}
    for item in result:
        ix, query, answer = item
        result_json[ix] = {
            'query': query,
            'answer': [{'text': a[0], 'score': float(a[1])} for a in answer]
        }
    json.dump(result_json, open(test_data_path+'_xgb_result.json', 'w'), indent=2, ensure_ascii=False)
    return top_1_cnt/total_cnt, top_3_cnt/total_cnt, top_5_cnt/total_cnt, top_10_cnt/total_cnt


if __name__ == "__main__":
    test_data_path = join(PATH_ARGS.DATA_DIR, 'baseline_dataset', 'dev.tsv')
    print('------basic dev dataset------')
    print(evaluate(test_data_path))

    test_data_path = join(PATH_ARGS.DATA_DIR, 'baseline_dataset', 'eda_dev.tsv')
    print('------eda dev dataset------')
    print(evaluate(test_data_path))

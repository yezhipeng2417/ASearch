import os, sys
join = os.path.join
dirname = os.path.dirname
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from config import *
import pickle as pkl
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
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

def evaluate():
    pass

if __name__ == "__main__":
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

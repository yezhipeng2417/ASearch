from copy import deepcopy as copy
from tabnanny import verbose

import pickle as pkl
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split  
from copy import deepcopy as copy 


RANDOM_STATE = 0

def load_dataset(data_path):
    dataset = open(data_path, 'r').readlines()
    x_processed_data = []
    y_processed_data = []
    
    for line in dataset:
        line = [float(i) for i in line.strip().split()]
        label = int(line[-1])
        features = line[:-1]
        x_processed_data.append(copy(features))
        y_processed_data.append(copy(label))
    X_train, X_test, y_train, y_test = train_test_split(x_processed_data, y_processed_data, test_size=0.25, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test

model = XGBClassifier()
x_train, x_test, y_train, y_test = load_dataset('/mnt/ASearch/ASearch/rerank/data/LCSTS_new/dataset.txt')
model.fit(x_train, y_train, verbose=True)
pkl.dump(model, open('/mnt/ASearch/ASearch/rerank/model_file/xgb.pkl', "wb"))
model = pkl.load(open('/mnt/ASearch/ASearch/rerank/model_file/xgb.pkl', "rb"))
y_pred = model.predict(x_test)
f1_report = classification_report(y_test, y_pred, target_names=['0', '1'])
print(f1_report)
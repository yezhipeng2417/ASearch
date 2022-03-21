import numpy as np
import pickle as pkl
from xgboost import XGBClassifier

model = pkl.load(open('/mnt/ASearch/ASearch/rerank/model_file/xgb.pkl', "rb"))
def predict(batch_features):
    y_pred = model.predict_proba(batch_features)
    print(y_pred)
    y_pred = [i[1] for i in y_pred]
    return y_pred

# y = predict([[0.0, 0.3, 0.8, 3.408934632675149e-05]])
# print(y)
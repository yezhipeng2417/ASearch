import os
import sys

join = os.path.join
dirname = os.path.dirname

sys.path.append(join(dirname(__file__), '..'))

import numpy as np
from src.text_simlirity import Text_Similarity, CTR_CQR_Score, Simbert_Score

ts = Text_Similarity()
def extract_feature(query, document, title=None):
    if title is None:
        document = [s for p in document.split('\n') for s in p.split('。') if s!= '']
        scores = []
        related_q_d_num = 0
        for line in document:
            line += '。'
            text_score, embed_score = ts.seprate_similarity(query, line)
            scores.append([text_score, embed_score, text_score + embed_score])
            line_score = text_score + embed_score
            # max_q_d_score = max(max_q_d_score, line_score)
            if line_score >= 0.7:
                related_q_d_num += 1
        related_q_d_num_score = related_q_d_num / len(document)
        scores.sort(key=lambda x: x[-1], reverse=True)
        scores = scores[:3]
        features = [len(scores)]
        if len(scores) < 3:
            scores = scores + (3-len(scores))*[[-1, -1, -2]]
        # print(scores)
        features = features + list(np.array(scores).flat) + [related_q_d_num, related_q_d_num_score, len(document)]
        # print(features)
        return features
    
        
# extract_feature(query='宁夏实现因私出国（境）护照跨市办理', document='昨日，记者从自治区公安厅获悉，自12月1日，我区居民可在区内五市（银川市、吴忠市、石嘴山市、中卫市、固原市）通办因私出国（境）证件以来，全区跨市就近办证人员已达2000人次。')
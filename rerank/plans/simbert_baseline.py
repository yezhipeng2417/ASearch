import os
import sys
join = os.path.join
dirname = os.path.dirname
sys.path.append(join(dirname(__file__), '..'))
import pandas as pd
from config import *
from tqdm import trange, tqdm
from copy import deepcopy as copy
from src.text_simlirity_batch import Cos_Similarity
from model.simbert import simbert_vec, simbert_single_vec


def read_data(data_path):
    data = pd.read_csv(data_path, sep='\t', names=['A', 'B', 'label'], encoding='gbk')
    return data

def get_all_vecs(data, batch_size):
    sents_A = data.A.tolist()
    embeds_A = []
    for inx in trange(0, len(sents_A), batch_size, desc='embed_A'):
        temp_batch = sents_A[inx: inx+batch_size]
        temp_batch_embeds = simbert_vec(temp_batch)
        embeds_A.extend(temp_batch_embeds)
    data['embed_A'] = embeds_A

    sents_B = data.B.tolist()
    embeds_B = []
    for inx in trange(0, len(sents_B), batch_size, desc='embed_B'):
        temp_batch = sents_B[inx: inx+batch_size]
        temp_batch_embeds = simbert_vec(temp_batch)
        embeds_B.extend(temp_batch_embeds)
    data['embed_B'] = embeds_B
    return data

def find_most_similar(embeds_A, embeds_B):
    result = {}
    cos_sim = Cos_Similarity()
    for ix, embed in enumerate(tqdm(embeds_A, desc='calculate cosine similarity')):
        temp_score = cos_sim.cos_similarity_matrix(embed, embeds_B)
        temp_score = list(enumerate(temp_score))
        temp_score = sorted(temp_score, key=lambda x: x[-1], reverse=True)[:10]
        result[ix] = {
                'scores': temp_score,
                'ids': [i[0] for i in temp_score]
            }
    return result

def evaluate(result):
    total_cnt = len(result)
    top_1_cnt = 0
    top_3_cnt = 0
    top_5_cnt = 0
    top_10_cnt = 0
    for k, v in result.items():
        if k in v['ids'][:1]:
            top_1_cnt += 1
            top_3_cnt += 1
            top_5_cnt += 1
            top_10_cnt += 1
        elif k in v['ids'][:3]:
            top_3_cnt += 1
            top_5_cnt += 1
            top_10_cnt += 1
        elif k in v['ids'][:5]:
            top_5_cnt += 1
            top_10_cnt += 1
        elif k in v['ids'][:10]:
            top_10_cnt += 1
        else:
            continue
    return top_1_cnt/total_cnt, top_3_cnt/total_cnt, top_5_cnt/total_cnt, top_10_cnt/total_cnt


if __name__ == "__main__":
    test_data_path = join(PATH_ARGS.DATA_DIR, 'baseline_dataset', 'eda_dev.tsv')
    data = read_data(test_data_path)
    data = get_all_vecs(data, 128)
    result = find_most_similar(data.embed_A, data.embed_B)
    _1, _3, _5, _10 = evaluate(result)
    print(_1, _3, _5, _10)

# 2000 basic 测试数据集
# top1 0.82 
# top3 0.94 
# top5 0.95 
# top10 0.96

# 2000 eda 测试数据集
# top1 0.5345
# top3 0.889
# top5 0.9365
# top10 0.9585
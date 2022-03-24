import os
import sys

join = os.path.join
dirname = os.path.dirname
sys.path.append(join(dirname(__file__), '..'))

import numpy as np
from tqdm import tqdm
from config import BASIC_ARGS
from copy import deepcopy as copy 
from src.text_simlirity_batch import Text_Similarity

bts = BASIC_ARGS.BATCH_SIZE

ts = Text_Similarity()
def extract_feature(sents_a, sents_b):
    print('lengths', len(sents_a), len(sents_b))
    range_a = list(range(0, len(sents_a), bts))
    range_b = list(range(0, len(sents_b), bts))
    pbar = tqdm(total=len(range_a)*len(range_b), desc='extract features')
    result = []
    for part_a_ix in range_a:
        part_a = sents_a[part_a_ix: part_a_ix+bts]
        temp_result = []
        for part_b_ix in range_b:
            part_b = sents_b[part_b_ix: part_b_ix+bts]
            sims = ts.seprate_similarity(part_a, part_b)
            # print(sims)
            if temp_result == []:
                temp_result.extend(sims)
            else:
                temp_result = [i+j for i, j in zip(temp_result, sims)]
            pbar.update(1)
        # print('#', temp_result)
        result.extend(copy(temp_result))
    return result

# if __name__ == "__main__":
#     a, b = ['你知道如何祛斑', '今天天气真糟糕', '你知道如何祛斑', '今天天气真糟糕'], ['你知道如何祛斑吗哈哈哈', '今天天气真好啊', '今天天天气一点也不好']
#     a, b = [*a*10], [*b*10]
#     print(len(a), len(b))
#     result = extract_feature(a, b)
#     print(len(result), len(result[0]))

    # print(result)

# extract_feature(query='宁夏实现因私出国（境）护照跨市办理', document='昨日，记者从自治区公安厅获悉，自12月1日，我区居民可在区内五市（银川市、吴忠市、石嘴山市、中卫市、固原市）通办因私出国（境）证件以来，全区跨市就近办证人员已达2000人次。')
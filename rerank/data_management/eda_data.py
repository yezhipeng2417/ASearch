import json
import jieba
import random
import requests
import synonyms
from time import time
from tqdm import trange
from copy import deepcopy as copy
from nltk.corpus import wordnet as wn
random.seed(0)

########################################################################
# 同义词替换
# 随机替换一个语句中的n个单词为其同义词
########################################################################
def synonym_replacement(words, entity_words, n):
    new_words = copy(words)
    random.shuffle(new_words)

    num_replaced = 0  
    for random_word in new_words: # 将entity word过滤掉，这些要求不被替换
        if random_word not in entity_words:
            synonyms = get_synonyms(random_word, 0.7) # score>=0.7的同义词候选列表
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in words]
                
                num_replaced += 1
            
            if num_replaced >= n: 
                break

    return new_words

def get_synonyms(word, min_confidence=0.7):
    nearby_words = synonyms.nearby(word) # 返回一个元组，元组中包含两项：([nearby_words], [nearby_words_score])；words按score排序，score在0-1之间
    result = []
    for name, score in zip(nearby_words[0][1:], nearby_words[1][1:]):
        if score >= min_confidence:
            result.append(name)
    try:
        for synset in wn.synsets(word, lang='cmn'):
            for lemma in synset.lemma_names('cmn'):
                result.append(lemma)
    except:
        pass
    return result

########################################################################
# 随机插入
# 随机在语句中插入n个词
########################################################################
def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0    
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = random.choice(synonyms)
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

########################################################################
def run_eda(sentences, entities=[], alpha_sr=0.2, alpha_ri=0.2, alpha_rs=0.2, num_aug=10):
    '''
    entity是不希望被替换的词的列表
    alpha_sr: 百分之多少的词被替换
    '''
    result = {}
    for sentence, entity in zip(sentences, entities):
        # print(sentence, entity)
        # add jieba dictionary
        if len(entity) > 0: # 词加入词典
            for item in entity:
                jieba.add_word(item)

        words = jieba.lcut(sentence)
        num_words = len(words)

        augmented_sentences = set()
        num_new_per_technique = int(num_aug/3)+1
        n_sr = max(1, int(alpha_sr * num_words)) # 至少一个词被替换
        n_ri = max(1, int(alpha_ri * num_words))
        n_rs = max(1, int(alpha_rs * num_words))
        
        # 同义词替换
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, entity, n_sr)
            augmented_sentences.add(''.join(a_words))
        
        #随机插入ri
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n_ri)
            augmented_sentences.add(''.join(a_words))
        
        #随机交换rs
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n_rs)
            augmented_sentences.add(''.join(a_words))

        result[sentence] = list(augmented_sentences)[: num_aug]
    return result


# if __name__ == '__main__':
#     print(run_eda('计算机技术的发展极大方便了人们的生活', []))

# def get_eda_data(texts, n=3):
#     url = "http://139.217.99.187:80/eda"
#     result = []
#     batch_size = 5
#     for i in trange(0, len(texts), batch_size, desc='request eda'):
#         temp_texts = texts[i: i+batch_size]
#         payload = json.dumps({
#         "traceid": f"eda+{str(time())}",
#         "text": texts,
#         "no_change_list": [[]]*len(temp_texts),
#         "topk": n
#         })
#         headers = {
#         'Content-Type': 'application/json'
#         }

#         response = requests.request("POST", url, headers=headers, data=payload).json()
#         for v in response['results'].values():
#             result.append(copy(v))
#     return result
    
def get_eda_data(texts, n=5):
    result = []
    batch_size = 10
    for i in trange(0, len(texts), batch_size, desc='request eda'):
        temp_texts = texts[i: i+batch_size]
        results = run_eda(temp_texts, entities=[[]]*len(temp_texts))
        for v in results.values():
            result.append(copy(v))
    return result

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import BASIC_ARGS
import jieba
import numpy as np
from tqdm import tqdm, trange
from copy import deepcopy as copy
from model.simbert import simbert_single_vec, simbert_vec


class Cos_Similarity:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def norm_2_vector(vector):
        return np.linalg.norm(vector)
    
    @staticmethod
    def norm_2_matrix(matrix):
        matrix = np.array(matrix)
        matrix = matrix.T
        return np.linalg.norm(matrix, axis=0)
    
    def cos_similarity(self, vec_a, vec_b):
        a_norm = self.norm_2_vector(vec_a)
        b_norm = self.norm_2_vector(vec_b)
        cos = np.dot(vec_a, vec_b) / (a_norm*b_norm)
        return cos

    def cos_similarity_matrix_vec_mat(self, vec_a, mat_b):
        mat_b = np.array([np.array(i) for i in mat_b]) # to 2-dim np.array
        mat_b_t = mat_b.T # transpose
        dot_product = np.dot(vec_a, mat_b_t) # (1, embed_dim) * (embed_dim, batch_size) -> (1, batch_size)
        vec_a_2_norm_2 = self.norm_2_vector(vec_a) # number
        mat_b_2_norm_2 = self.norm_2_matrix(mat_b) # (1, batch_size)
        cos_sim = dot_product / (vec_a_2_norm_2*mat_b_2_norm_2) # (1, batch_size)
        return cos_sim

    def cos_similarity_matrix(self, mat_a, mat_b):
        mat_b = np.array([np.array(i) for i in mat_b]) # to 2-dim np.array
        mat_b_t = mat_b.T # transpose
        dot_product = np.dot(mat_a, mat_b_t) # (batch_size1, embed_dim) * (embed_dim, batch_size2) -> (batch_size1, batch_size2)
        mat_a_2_norm_2 = self.norm_2_matrix(mat_a) # (batch_size1, 1)
        mat_b_2_norm_2 = self.norm_2_matrix(mat_b) # (1, batch_size2)
        mat_a_2_norm_2_mat = np.array([mat_a_2_norm_2]*len(mat_b_2_norm_2)).T # (batch_size1, batch_size2)
        mat_b_2_norm_2_diag = np.diagflat(mat_b_2_norm_2) # (batch_size2, batch_size2) 
        cos_sim = dot_product / np.dot(mat_a_2_norm_2_mat, mat_b_2_norm_2_diag) # (batch_size1, batch_size2)
        return cos_sim


class Text_Similarity:
    def __init__(self) -> None:
        self.cs = Cos_Similarity()
    
    def calculate_sentence_info(self, sentences):
        sent2word_idx = {}
        words = []
        for ix, sent in enumerate(sentences):
            temp_words = jieba.lcut(sent)
            words_cnt = len(temp_words)
            words_index = len(words)
            sent2word_idx[ix] = list(range(words_index, words_index+words_cnt))
            words.extend(temp_words)
        words.extend(sentences)

        embeddings = []
        for i in trange(0, len(words), BASIC_ARGS.BATCH_SIZE, desc='calculate vector'):
            temp_words = words[i: i+BASIC_ARGS.BATCH_SIZE]
            temp_embeddings = simbert_vec(temp_words)
            embeddings.extend(temp_embeddings)
        
        sent_embed, word_embed = embeddings[-len(sentences):], embeddings[:-len(sentences)]
        sent_word_cos_sim = self.cs.cos_similarity_matrix(sent_embed, word_embed)
        result = {}
        for sent_id, word_ids in sent2word_idx.items():
            temp_words_info = {}
            for word_id in word_ids:
                word = words[word_id]
                word_score = sent_word_cos_sim[sent_id][word_id]
                if word in temp_words_info:
                    temp_words_info[word] += word_score
                else:
                    temp_words_info[word] = word_score
            result[sentences[sent_id]] = {
                'words_info': copy(temp_words_info),
                'embedding': sent_embed[sent_id]
            }
        return result

    @staticmethod
    def _ctr_cqr_score(query, text):
        sent_a_words = set(query['words_info'].keys())
        sent_b_words = set(text['words_info'].keys())

        inter_sec = list(sent_a_words.intersection(sent_b_words))
        ctr = sum([text['words_info'][i] for i in inter_sec]) / sum(list(text['words_info'].values()))
        cqr = sum([text['words_info'][i] for i in inter_sec]) / sum(list(query['words_info'].values()))
        # ctr_cqr = ctr * cqr
        return ctr, cqr
    
    def _simbert_score(self, querys, texts):
        querys = [i['embedding'] for i in querys]
        texts = [i['embedding'] for i in texts]
        score = self.cs.cos_similarity_matrix(querys, texts)
        return score
    
    def seprate_similarity(self, sents_a, sents_b):
        sents_a_info = self.calculate_sentence_info(sents_a)
        sents_b_info = self.calculate_sentence_info(sents_b)
        string_scores, semantics_scores = [], []
        
        for a in sents_a_info.values():
            temp_string_scores = []
            for b in sents_b_info.values():
                temp_string_score = self._ctr_cqr_score(a, b)
                temp_string_scores.append(copy(temp_string_score))
            string_scores.append(copy(temp_string_scores))
        semantics_scores = self._simbert_score(sents_a_info.values(), sents_b_info.values())
        result = [[[*string_score, semantics_score] for string_score, semantics_score in zip(string_score_line, semantics_score_line)] 
                                                    for string_score_line, semantics_score_line in zip(string_scores, semantics_scores)]
        return result


# if __name__ == "__main__":
    # mat_a, mat_b = [[1,1,1], [1,2,4]], [[2,2,2], [1,2,2], [3,4,5], [2,2,2]]
    # cs = Cos_Similarity()
    # print('------mat*mat------')
    # print(cs.cos_similarity_matrix(mat_a, mat_b))
    
    # print('------vec*mat------')
    # for i in mat_a:
    #     print(cs.cos_similarity_matrix_vec_mat(i, mat_b))
    
    # print('------vec*vec------')
    # for i in mat_a:
    #     res = []
    #     for j in mat_b:
    #         res.append(cs.cos_similarity(i, j))
    #     print(res)
    # from time import time
    # ts = Text_Similarity()
    # a, b = ['你知道如何祛斑', '今天天气真糟糕', '你知道如何祛斑', '今天天气真糟糕'], ['你知道如何祛斑吗哈哈哈', '今天天气真好啊', '今天天天气一点也不好']
    # a = [*a*10]
    # b = [*b*10]
    # print(a)
    # print(b)
    # start_time = time()
    # ts.seprate_similarity(a, b)
    # print(time()-start_time)
    
    # start_time = time()
    # for i in a:
    #     for j in b:
    #         ts.seprate_similarity([i], [j])
    # print(time()-start_time)

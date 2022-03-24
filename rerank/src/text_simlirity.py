import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../model'))

import jieba
import numpy as np
from simbert import *
from model.simbert import simbert_single_vec


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
    
    def cos_similarity_matrix(self, vec_a, mat_b):
        dot_product = np.dot(vec_a, mat_b)
        vec_a_2_norm_2 = self.norm_2_vector(vec_a)
        mat_b_2_norm_2 = self.norm_2_matrix(mat_b)
        cos_sim = dot_product / (vec_a_2_norm_2*mat_b_2_norm_2)
        return cos_sim

class CTR_CQR_Score:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _ctr_cqr_score(query_words, text_words):
        query = set(query_words.keys())
        text = set(text_words.keys())
        inter_sec = list(set(query).intersection(set(text)))
        ctr = sum([text_words[i] for i in inter_sec]) / sum(list(text_words.values()))
        cqr = sum([text_words[i] for i in inter_sec]) / sum(list(query_words.values()))
        ctr_cqr = ctr * cqr
        return ctr_cqr

    def word_weight(self, sentence):
        sen_vec = simbert_single_vec(sentence)
        words = jieba.lcut(sentence)
        words_dict = {}
        for w in words:
            if w in words_dict:
                words_dict[w] += 1
            else:
                words_dict[w] = 1
        for w in words_dict.keys():
            vec = simbert_single_vec(w)
            w_score = self.norm_weight(vec, sen_vec)
            words_dict[w] *= w_score
        return words_dict

    @staticmethod
    def norm_2(vector):
        return np.linalg.norm(vector)

    def cos_similarity(self, vec_a, vec_b):
        a_norm = self.norm_2(vec_a)
        b_norm = self.norm_2(vec_b)
        cos = np.dot(vec_a, vec_b) / (a_norm*b_norm)
        return cos

    def norm_weight(self, word_vec, sen_vec):
        word_norm = self.norm_2(word_vec)
        sen_norm = self.norm_2(sen_vec)
        cos_sim = self.cos_similarity(word_vec, sen_vec)
        weight = word_norm * cos_sim / sen_norm
        return weight
    
    def get_ctr_cqr_score(self, query, text):
        query_word_weight = self.word_weight(query)
        text_word_weight = self.word_weight(text)
        score = self._ctr_cqr_score(query_word_weight, text_word_weight)
        return score

class Simbert_Score:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def norm_2(vector):
        return np.linalg.norm(vector)
    
    @staticmethod
    def norm_2_matrix(matrix):
        matrix = np.array(matrix)
        matrix = matrix.T
        return np.linalg.norm(matrix, axis=0)

    def cos_similarity(self, sen_a, sen_b):
        vec_a = simbert_single_vec(sen_a)
        vec_b = simbert_single_vec(sen_b)
        a_norm = self.norm_2(vec_a, )
        b_norm = self.norm_2(vec_b)
        cos = np.dot(vec_a, vec_b) / (a_norm*b_norm)
        return cos
    
    def cos_similarity_batch(self, query, batch):
        pass
    
    def simbert_score_batch(self, batch):
        pass

class Text_Similarity:
    def __init__(self) -> None:
        self.ts = CTR_CQR_Score()
        self.simbert = Simbert_Score()
    
    def similarity(self, sen_a, sen_b):
        text_score, embed_score = self.seprate_similarity(sen_a, sen_b)
        return (text_score + embed_score)/2
    
    def seprate_similarity(self, sen_a, sen_b):
        text_score = self.ts.get_ctr_cqr_score(sen_a, sen_b)
        embed_score = self.simbert.cos_similarity(sen_a, sen_b)
        return text_score, embed_score

    def q_t_d_sim(self, query, title, doc):
        pass
    
    def q_d_sim(self, query, title, doc):
        doc = [s for p in doc.split('\n') for s in p.split('。') if s!= '']
        pass


# ts = Text_Similarity()
# print(ts.similarity('我喜欢吃西红柿', '我不喜欢吃西红柿'))

# cs = Cos_Similarity()
# res = cs.cos_similarity_matrix([1,2,3], [[2,3,4], [4, 6, 8], [4, 8, 12]])
# print(res)

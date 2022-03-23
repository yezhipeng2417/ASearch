import os, sys
from unittest import main
sys.path.append(os.path.join(os.path.dirname(__file__), '../model'))

import jieba
import numpy as np
from simbert import *
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
    
    def cos_similarity_matrix(self, vec_a, mat_b):
        mat_b = np.array([np.array(i) for i in mat_b]) # to 2-dim np.array
        mat_b_t = mat_b.T # transpose
        dot_product = np.dot(vec_a, mat_b_t) # (1, embed_dim) * (embed_dim, batch_size) -> (1, batch_size)
        vec_a_2_norm_2 = self.norm_2_vector(vec_a) # number
        mat_b_2_norm_2 = self.norm_2_matrix(mat_b) # (1, batch_size)
        cos_sim = dot_product / (vec_a_2_norm_2*mat_b_2_norm_2) # (1, batch_size)
        return cos_sim

class CTR_CQR_Score:
    def __init__(self) -> None:
        pass
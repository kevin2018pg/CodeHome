# -*- coding: utf-8 -*-
# @Time    : 2022/6/30 16:17
# @Author  : kevin
# @Version : python 3.7
# @Desc    : utils

import gc
import time

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy import linalg
from scipy.special import iv
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm


def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(cols):
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


class ProNE():
    def __init__(self, G, emb_size=100, step=10, theta=0.5, mu=0.2, n_iter=5, random_state=2021):
        self.G = G
        self.emb_size = emb_size
        self.G = self.G.to_undirected()
        self.node_number = self.G.number_of_nodes()
        self.random_state = random_state
        self.step = step
        self.theta = theta
        self.mu = mu
        self.n_iter = n_iter

        mat = scipy.sparse.lil_matrix((self.node_number, self.node_number))

        for e in tqdm(self.G.edges()):
            if e[0] != e[1]:
                mat[int(e[0]), int(e[1])] = 1
                mat[int(e[1]), int(e[0])] = 1
        self.mat = scipy.sparse.csr_matrix(mat)
        print(mat.shape)

    def get_embedding_rand(self, matrix):
        # Sparse randomized tSVD for fast embedding
        t1 = time.time()
        l = matrix.shape[0]
        smat = scipy.sparse.csc_matrix(matrix)  # convert to sparse CSC format
        print('svd sparse', smat.data.shape[0] * 1.0 / l ** 2)
        U, Sigma, VT = randomized_svd(smat, n_components=self.emb_size, n_iter=self.n_iter,
                                      random_state=self.random_state)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        print('sparsesvd time', time.time() - t1)
        return U

    def get_embedding_dense(self, matrix, emb_size):
        # get dense embedding via SVD
        t1 = time.time()
        U, s, Vh = linalg.svd(matrix, full_matrices=False,
                              check_finite=False, overwrite_a=True)
        U = np.array(U)
        U = U[:, :emb_size]
        s = s[:emb_size]
        s = np.sqrt(s)
        U = U * s
        U = preprocessing.normalize(U, "l2")
        print('densesvd time', time.time() - t1)
        return U

    def fit(self, tran, mask):
        # Network Embedding as Sparse Matrix Factorization
        t1 = time.time()
        l1 = 0.75
        C1 = preprocessing.normalize(tran, "l1")
        neg = np.array(C1.sum(axis=0))[0] ** l1

        neg = neg / neg.sum()

        neg = scipy.sparse.diags(neg, format="csr")
        neg = mask.dot(neg)
        print("neg", time.time() - t1)

        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1

        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)

        C1 -= neg
        F = C1
        features_matrix = self.get_embedding_rand(F)
        return features_matrix

    def chebyshev_gaussian(self, A, a, order=10, mu=0.5, s=0.5):
        # NE Enhancement via Spectral Propagation
        print('Chebyshev Series -----------------')
        t1 = time.time()

        if order == 1:
            return a

        A = sp.eye(self.node_number) + A
        DA = preprocessing.normalize(A, norm='l1')
        L = sp.eye(self.node_number) - DA

        M = L - mu * sp.eye(self.node_number)

        Lx0 = a
        Lx1 = M.dot(a)
        Lx1 = 0.5 * M.dot(Lx1) - a

        conv = iv(0, s) * Lx0
        conv -= 2 * iv(1, s) * Lx1
        for i in range(2, order):
            Lx2 = M.dot(Lx1)
            Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
            #         Lx2 = 2*L.dot(Lx1) - Lx0
            if i % 2 == 0:
                conv += 2 * iv(i, s) * Lx2
            else:
                conv -= 2 * iv(i, s) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
            del Lx2
            print('Bessell time', i, time.time() - t1)
        mm = A.dot(a - conv)
        self.embeddings = self.get_embedding_dense(mm, self.emb_size)
        return self.embeddings

    def transform(self):
        if self.embeddings is None:
            print("Embedding is not train")
            return {}
        self.embeddings = pd.DataFrame(self.embeddings)
        self.embeddings.columns = ['ProNE_Emb_{}'.format(i) for i in range(len(self.embeddings.columns))]
        self.embeddings = self.embeddings.reset_index().rename(columns={'index': 'nodes'}).sort_values(by=['nodes'],
                                                                                                       ascending=True).reset_index(
            drop=True)
        return self.embeddings

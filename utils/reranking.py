# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Dec, 25 May 2019 20:29:09
# Faster version for kesci ReID challenge

# @author: luohao
# """

# """
# CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
# url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
# Matlab version: https://github.com/zhunzhong07/person-re-ranking
# """

# """
# API

# probFea: all feature vectors of the query set (torch tensor)
# probFea: all feature vectors of the gallery set (torch tensor)
# k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
# MemorySave: set to 'True' when using MemorySave mode
# Minibatch: avaliable when 'MemorySave' is 'True'
# """

# Save memory version

import numpy as np
import torch
import time
import gc
from tqdm import tqdm



def euclidean_distance(qf, gf):

    m = qf.shape[0]
    n = gf.shape[0]

    # dist_mat = torch.pow(qf,2).sum(dim=1, keepdim=True).expand(m,n) +\
    #     torch.pow(gf,2).sum(dim=1, keepdim=True).expand(n,m).t()
    # dist_mat.addmm_(1,-2,qf,gf.t())

    # for L2-norm feature
    dist_mat = 2 - 2 * torch.matmul(qf, gf.t())
    return dist_mat


def batch_euclidean_distance(qf, gf, N=6000):
    m = qf.shape[0]
    n = gf.shape[0]

    dist_mat = []
    for j in range(n // N + 1):
        temp_gf = gf[j * N:j * N + N]
        temp_qd = []
        for i in range(m // N + 1):
            temp_qf = qf[i * N:i * N + N]
            temp_d = euclidean_distance(temp_qf, temp_gf)
            temp_qd.append(temp_d)
        temp_qd = torch.cat(temp_qd, dim=0)
        temp_qd = temp_qd / (torch.max(temp_qd, dim=0)[0])
        dist_mat.append(temp_qd.t().cpu())
    del temp_qd
    del temp_gf
    del temp_qf
    del temp_d
    torch.cuda.empty_cache()  # empty GPU memory
    dist_mat = torch.cat(dist_mat, dim=0)
    return dist_mat


# 将topK排序放到GPU里运算，并且只返回k1+1个结果
# Compute TopK in GPU and return (k1+1) results
def batch_torch_topk(qf, gf, k1, N=6000):
    m = qf.shape[0]
    n = gf.shape[0]

    dist_mat = []
    initial_rank = []
    for j in range(n // N + 1):
        temp_gf = gf[j * N:j * N + N]
        temp_qd = []
        for i in range(m // N + 1):
            temp_qf = qf[i * N:i * N + N]
            temp_d = euclidean_distance(temp_qf, temp_gf)
            temp_qd.append(temp_d)
        temp_qd = torch.cat(temp_qd, dim=0)
        temp_qd = temp_qd / (torch.max(temp_qd, dim=0)[0])
        temp_qd = temp_qd.t()
        initial_rank.append(torch.topk(temp_qd, k=k1, dim=1, largest=False, sorted=True)[1])

    del temp_qd
    del temp_gf
    del temp_qf
    del temp_d
    torch.cuda.empty_cache()  # empty GPU memory
    initial_rank = torch.cat(initial_rank, dim=0).cpu().numpy()
    return initial_rank


def batch_v(feat, R, all_num):
    V = np.zeros((all_num, all_num), dtype=np.float32)
    m = feat.shape[0]
    for i in tqdm(range(m)):
        temp_gf = feat[i].unsqueeze(0)
        # temp_qd = []
        temp_qd = euclidean_distance(temp_gf, feat)
        temp_qd = temp_qd / (torch.max(temp_qd))
        temp_qd = temp_qd.squeeze()
        temp_qd = temp_qd[R[i]]
        weight = torch.exp(-temp_qd)
        weight = (weight / torch.sum(weight)).cpu().numpy()
        V[i, R[i]] = weight.astype(np.float32)
    return V


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def re_ranking(probFea, galFea, k1, k2, lambda_value):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    t1 = time.time()
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    feat = torch.cat([probFea, galFea]).cuda()
    initial_rank = batch_torch_topk(feat, feat, k1 + 1, N=6000)
    # del feat
    del probFea
    del galFea
    torch.cuda.empty_cache()  # empty GPU memory
    gc.collect()  # empty memory
    print('Using totally {:.2f}s to compute initial_rank'.format(time.time() - t1))
    print('starting re_ranking')

    R = []
    for i in tqdm(range(all_num)):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        R.append(k_reciprocal_expansion_index)

    gc.collect()  # empty memory
    print('Using totally {:.2f}S to compute R'.format(time.time() - t1))
    V = batch_v(feat, R, all_num)
    del R
    gc.collect()  # empty memory
    print('Using totally {:.2f}S to compute V-1'.format(time.time() - t1))
    initial_rank = initial_rank[:, :k2]

    ### 下面这个版本速度更快
    ### Faster version
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank

    ### 下面这个版本更省内存(约40%)，但是更慢
    ### Low-memory version
    '''gc.collect()  # empty memory
    N = 2000
    for j in range(all_num // N + 1):

        if k2 != 1:
            V_qe = np.zeros_like(V[:, j * N:j * N + N], dtype=np.float32)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i], j * N:j * N + N], axis=0)
            V[:, j * N:j * N + N] = V_qe
            del V_qe
    del initial_rank'''

    gc.collect()  # empty memory
    print('Using totally {:.2f}S to compute V-2'.format(time.time() - t1))
    invIndex = []

    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])
    print('Using totally {:.2f}S to compute invIndex'.format(time.time() - t1))

    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)
    for i in tqdm(range(query_num)):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)
    del V
    gc.collect()  # empty memory
    original_dist = batch_euclidean_distance(feat, feat[:query_num, :]).numpy()
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    # print(jaccard_dist)
    del original_dist

    del jaccard_dist

    final_dist = final_dist[:query_num, query_num:]
    print(final_dist)
    print('Using totally {:.2f}S to compute final_distance'.format(time.time() - t1))
    return final_dist

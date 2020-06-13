import numpy as np
import os
import json


base_dir = '../model/'

query_1 = np.load(base_dir + '1229-a-1/query_path_1.npy')
gallery_1 = np.load(base_dir + '1229-a-1/gallery_path_1.npy')
print(len(query_1), len(gallery_1))
query_2 = np.load(base_dir + '1229-a-1/query_path_2.npy')
gallery_2 = np.load(base_dir + '1229-a-1/gallery_path_2.npy')
print(len(query_2), len(gallery_2))

distmat_1 = np.load(base_dir + '1229-a-1/dist_mat_1.npy')
distmat_1 += np.load(base_dir + '0102-b-2/dist_mat_1.npy')
distmat_1 += np.load(base_dir + '0102-a-se-1/dist_mat_1.npy')

print(distmat_1.shape)
indexes = np.argsort(distmat_1, axis=1)

res_1 = {}
for idx, index in enumerate(indexes):
    query = os.path.basename(query_1[idx])
    gallery = [os.path.basename(i) for i in gallery_1[index][:200].tolist()]
    res_1[query] = gallery


distmat_2 = np.load(base_dir + '1229-a-1/dist_mat_2.npy')
distmat_2 += np.load(base_dir + '0102-b-2/dist_mat_2.npy')
distmat_2 += np.load(base_dir + '0102-a-se-1/dist_mat_2.npy')

print(distmat_2.shape)

indexes = np.argsort(distmat_2, axis=1)

res_2 = {}
for idx, index in enumerate(indexes):
    query = os.path.basename(query_2[idx])
    gallery = [os.path.basename(i) for i in gallery_2[index][:200].tolist()]
    res_2[query] = gallery

data = dict()
for k, v in res_1.items():
    data[k] = v
for k, v in res_2.items():
    data[k] = v

save_path = 'submit_final.json'
print("Writing to {}".format(save_path))
json.dump(data, open(save_path, 'w'))

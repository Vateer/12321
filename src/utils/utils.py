import numpy as np
import torch
import random, copy
from sklearn.cluster import OPTICS
from typing import Iterable 
import torch.nn.functional as F

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def round_with_probability(num):
    decimal_part = num - int(num)
    if random.random() < decimal_part:
        return int(num) + 1
    else:
        return int(num)

def get_test_loader(args, test_datas, test_labels):
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(test_datas), torch.from_numpy(test_labels)),
        batch_size=len(test_labels), shuffle=False)
    return test_loader


# U是所有客户端
def calculating_adjacency(args, U): 
    clients_idxs = np.arange(args.clients)
    nclients = len(clients_idxs)
    
    sim_mat = np.zeros([nclients, nclients])
    for idx1 in range(nclients):
        for idx2 in range(nclients):
            #print(idx1)
            #print(U)
            #print(idx1)
            U1 = copy.deepcopy(U[clients_idxs[idx1]])
            U2 = copy.deepcopy(U[clients_idxs[idx2]])
            
            #sim_mat[idx1,idx2] = np.where(np.abs(U1.T@U2) > 1e-2)[0].shape[0]
            #sim_mat[idx1,idx2] = 10*np.linalg.norm(U1.T@U2 - np.eye(15), ord='fro')
            #sim_mat[idx1,idx2] = 100/np.pi*(np.sort(np.arccos(U1.T@U2).reshape(-1))[0:4]).sum()
            mul = np.clip(U1.T@U2 ,a_min =-1.0, a_max=1.0) # @表示矩阵乘法, 同时将结果限制在[-1,1]
            sim_mat[idx1,idx2] = np.min(np.arccos(mul))*180/np.pi #算最小的角度
           
    return sim_mat

def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

# linkage表示合并簇的时候如何计算相似度(也就是两个簇合并时如何计算与其他簇的距离)
def hierarchical_clustering(AA, thresh=1.5, linkage='maximum'):
    '''
    Hierarchical Clustering Algorithm. It is based on single linkage, finds the minimum element and merges
    rows and columns replacing the minimum elements. It is working on adjacency matrix. 
    
    :param: A (adjacency matrix), thresh (stopping threshold)
    :type: A (np.array), thresh (int)
    
    :return: clusters
    '''
    A = copy.deepcopy(AA)
    label_assg = {i: i for i in range(A.shape[0])} #初始化标签，一开始为自己
    
    step = 0
    while A.shape[0] > 1:
        np.fill_diagonal(A,-np.NINF) #对角线设置负无穷避免选到
        #print(f'step {step} \n {A}')
        step+=1
        ind=np.unravel_index(np.argmin(A, axis=None), A.shape) #寻找最小元素索引

        if A[ind[0],ind[1]]>thresh: #最小元素大于阈值，停止聚类
            print('Breaking HC')
            break
        else:
            np.fill_diagonal(A,0) #将对角线元素设置为0
            if linkage == 'maximum':
                Z=np.maximum(A[:,ind[0]], A[:,ind[1]]) #A[:,:,:] : 多维矩阵下每一个维度独立选择
            elif linkage == 'minimum':
                Z=np.minimum(A[:,ind[0]], A[:,ind[1]])
            elif linkage == 'average':
                Z= (A[:,ind[0]] + A[:,ind[1]])/2
            
            A[:,ind[0]]=Z
            A[:,ind[1]]=Z
            A[ind[0],:]=Z
            A[ind[1],:]=Z
            A = np.delete(A, (ind[1]), axis=0)
            A = np.delete(A, (ind[1]), axis=1)

            if type(label_assg[ind[0]]) == list: 
                label_assg[ind[0]].append(label_assg[ind[1]])
            else: 
                label_assg[ind[0]] = [label_assg[ind[0]], label_assg[ind[1]]] # 合并簇

            label_assg.pop(ind[1], None) #删掉簇

            temp = []
            for k,v in label_assg.items(): # 整理簇编号
                if k > ind[1]: 
                    kk = k-1
                    vv = v
                else: 
                    kk = k 
                    vv = v
                temp.append((kk,vv))

            label_assg = dict(temp) 

    clusters = []
    for k in label_assg.keys():
        if type(label_assg[k]) == list:
            clusters.append(list(flatten(label_assg[k])))
        elif type(label_assg[k]) == int: 
            clusters.append([label_assg[k]])
            
    return clusters

'''
metric='minkowski'是指在计算数据点之间的距离时使用的度量方法
在Scikit-learn中，可以用不同的metric参数值来指定不同的距离度量方法。除了Minkowski距离，还有以下一些选项：
'euclidean': 欧几里得距离，也就是我们通常说的直线距离，或者L2范数。
'manhattan': 曼哈顿距离，也就是城市块距离，或者L1范数。
'chebyshev': 切比雪夫距离，即两点之间的最大坐标差。
'cosine': 余弦距离，衡量两个向量之间的角度。
'''
def optics_clustering(AA, min_samples=5, xi=0.05, metric='minkowski'):
    '''
    OPTICS Clustering Algorithm. It is based on density, and does not require a preset number of clusters. 
    
    :param: A (adjacency matrix), min_samples (minimum number of samples in a cluster), xi (density threshold for defining a cluster), metric (distance metric)
    :type: A (np.array), min_samples (int), xi (float), metric (str)
    
    :return: clusters
    '''
    A = np.array(AA)
    clustering = OPTICS(min_samples=min_samples, xi=xi, metric=metric).fit(A)
    labels = clustering.labels_

    clusters = []
    for label in set(labels):
        clusters.append([i for i, x in enumerate(labels) if x == label])
    return clusters


def shannon_entropy(output):
    """
    Calculate the Shannon Entropy for the given model output (probabilities).
    """
    p = F.softmax(output, dim=1)
    log_p = torch.log(p)
    entropy = -torch.sum(p * log_p, dim=1)
    return entropy

def confidence(output):
    """
    Calculate the confidence of the model output using the maximum probability.
    """
    p = F.softmax(output, dim=1)
    max_prob, _ = torch.max(p, dim=1)
    return max_prob

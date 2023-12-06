import numpy as np
import scipy.sparse as sp
import torch
#这个函数为取K近邻函数

def myKNN(S, k, sigma=1.0):
    N = len(S)
    A = np.zeros((N,N))

    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours

        for j in neighbours_id: # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
            A[j][i] = A[i][j] # mutually
            if i == j:
                A[j][i] = A[i][j] = 0
    return A


def myKNNBoolean(S, k, sigma=1.0):
    N = len(S)
    A = np.zeros((N,N))

    for i in range(N):
        A[i][i]=1
        #生成索引
        dist_with_index = zip(S[i], range(N))
        #按照元组第一个元素排序
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])

        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours

        for j in neighbours_id: # xj is xi's neighbour

            A[i][j] = 1
            A[j][i] = A[i][j] # mutually
            # if i == j:
            #     A[j][i] = A[i][j] = 0

    dictA={}
    for i in range(N):
        j_list = []
        for j in range(N):
            if A[i][j] > 0:
                j_list.append(j)
        dictA.update({i:j_list})
    return A,dictA



            
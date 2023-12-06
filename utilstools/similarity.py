import numpy as np
#这个函数为距离函数集合
def euclidDistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res


def cosdDistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res

#求欧式距离矩阵
def calEuclidDistanceMatrix(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
    return S

#求欧式距离矩阵
def calEuclidDistanceMatrixSingle(X,X2,j):
    X = np.array(X)
    X2 = np.array(X2)
    S = np.zeros((len(X2)))
    for i in range(len(X2)):
        S[i] = 1.0 * euclidDistance(X[j], X2[i])
    return S

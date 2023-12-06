from scipy.sparse import csr_matrix
from torch import mean
from torch.nn.functional import cosine_similarity

from utilstools.knn import myKNN, myKNNBoolean
from utilstools.similarity import calEuclidDistanceMatrix
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp

from utilstools.utils2 import normalize, normalize_adj
#数据扰动函数集合

#data为数据集，k为k近邻数一般设置为3~15，alph为常系数一般设置为1~3
def createdata(data,k,alph):
    Similarity = calEuclidDistanceMatrix(data)
    # Similarity2 = calEuclidDistanceMatrix(np.transpose(data))
    data1=torch.zeros(data.size())
    data2=torch.zeros(data.size())
    # data3=torch.zeros(np.transpose(data).size())
    # data4=torch.zeros(np.transpose(data).size())
    Adjacent,_ = myKNNBoolean(Similarity, k)
    adj = Adjacent

    row, col = np.nonzero(adj)
    values = adj[row, col]
    csr_adjknn = csr_matrix((values, (row, col)), shape=adj.shape)
    # # a=normalize_adj2(adjknn + sp.eye(adjknn.shape[0]))
    adj = np.array(normalize_adj(csr_adjknn).tocsr().todense())

    np.save(f'./data/adjknnpubmed{int(k)}.npy', adj)

    # Adjacent2 = myKNN(Similarity2, k)
    for i in range(Adjacent.shape[0]):
        tmpdata = data[np.where(Adjacent[i,:]!=0)[0],:]
        tmp_std = torch.zeros(1,tmpdata.shape[1])
        for l in range(tmpdata.shape[1]):
            tmp_std[:,l]=torch.std(tmpdata[:,l])
        data1[i, :]=data[i,:] + alph * tmp_std*torch.ones(1,data.shape[1])
        data2[i, :] = data[i,:] - alph * tmp_std*torch.ones(1,data.shape[1])

    # for j in range(Adjacent2.shape[0]):
    #     tmpdata = np.transpose(data)[np.where(Adjacent2[j,:]!=0)[0],:]
    #     tmp_std = torch.zeros(1,data4.shape[1])
    #     for l in range(data4.shape[1]):
    #         tmp_std[:,l]=torch.std(tmpdata[:,l])
    #     data3[j, :]=np.transpose(data)[j,:] + alph * tmp_std*torch.ones(1,data3.shape[1])
    #     data4[j, :] = np.transpose(data)[j,:] - alph * tmp_std*torch.ones(1,data4.shape[1])
    return torch.FloatTensor(data1),torch.FloatTensor(data2)
            # ,torch.FloatTensor(np.transpose(data3)),torch.FloatTensor(np.transpose(data4)))

def sim(z1: torch.Tensor,lamda):
    z1 = F.normalize(z1)
    # z2 = F.normalize(z2)
    cos=torch.mm(z1, z1.t())
    # N=z1.shape[0]
    # for i in range(N-1):
    #     cos[i][i]=1
    #     for j in range(i+1,N):
    #         if cos[i][j] <= lamda:
    #             cos[i][j]=0
    #             cos[j][i] = 0
    #         else:
    #             cos[i][j] = 1
    #             cos[j][i] = 1
    cos[cos <= lamda] = 0
    cos[cos > lamda] = 1
    N=z1.shape[0]
    for i in range(N-1):
        cos[i][i]=1
    return cos

def createdatacos(data,lamda,alph):
    data1=torch.zeros(data.size()).to(torch.device("cuda"))
    data2=torch.zeros(data.size()).to(torch.device("cuda"))
    Adjacent = sim(data,lamda)

    adj = Adjacent.cpu().numpy()

    row, col = np.nonzero(adj)
    values = adj[row, col]
    csr_adjknn = csr_matrix((values, (row, col)), shape=adj.shape)
    # # a=normalize_adj2(adjknn + sp.eye(adjknn.shape[0]))
    adj = np.array(normalize_adj(csr_adjknn).tocsr().todense())

    np.save(f'./data/adjcospubmed{int(lamda*100)}.npy', adj)
    for i in range(Adjacent.shape[0]):
        tmpdata = data[torch.where(Adjacent[i,:]!=0)[0],:]
        if tmpdata.shape[0]>2:
            tmp_std = torch.zeros(1,tmpdata.shape[1]).to(torch.device("cuda"))
            for l in range(tmpdata.shape[1]):
                tmp_std[:,l]=torch.std(tmpdata[:,l])
            data1[i, :]=data[i, :] + alph * tmp_std*torch.ones(1,data.shape[1]).to(torch.device("cuda"))
            data2[i, :] = data[i,:] - alph * tmp_std*torch.ones(1,data.shape[1]).to(torch.device("cuda"))

    return data1,data2

def createdataaug(data,adj,alph):
    data=data.to(torch.device("cuda"))
    data1=torch.zeros(data.size()).to(torch.device("cuda"))
    data2=torch.zeros(data.size()).to(torch.device("cuda"))
    Adjacent = adj

    for i in range(Adjacent.shape[0]):
        tmpdata = data[torch.where(Adjacent[i,:]!=0)[0],:]
        if tmpdata.shape[0]>2:
            tmp_std = torch.zeros(1,tmpdata.shape[1]).to(torch.device("cuda"))
            for l in range(tmpdata.shape[1]):
                tmp_std[:,l]=torch.std(tmpdata[:,l])
            data1[i, :]=data[i, :] + alph * tmp_std*torch.ones(1,data.shape[1]).to(torch.device("cuda"))
            data2[i, :] = data[i,:] - alph * tmp_std*torch.ones(1,data.shape[1]).to(torch.device("cuda"))

    return data1,data2

def createdatacosraw(data,lamda,alph,adj):
    data1=torch.zeros(data.size()).to(torch.device("cuda"))
    data2=torch.zeros(data.size()).to(torch.device("cuda"))
    Adjacent = adj
    for i in range(Adjacent.shape[0]):
        tmpdata = data[torch.where(Adjacent[i,:]!=0)[0],:]
        if tmpdata.shape[0]>2:
            tmp_std = torch.zeros(1,tmpdata.shape[1]).to(torch.device("cuda"))
            for l in range(tmpdata.shape[1]):
                tmp_std[:,l]=torch.std(tmpdata[:,l])
            data1[i, :]=data[i, :] + alph * tmp_std*torch.ones(1,data.shape[1]).to(torch.device("cuda"))
            data2[i, :] = data[i,:] - alph * tmp_std*torch.ones(1,data.shape[1]).to(torch.device("cuda"))

    return data1,data2

def createdatacosdot(data,lamda,alph):
    data1=torch.zeros(data.size())
    data2=torch.zeros(data.size())
    Adjacent = sim(data,lamda)
    for i in range(Adjacent.shape[0]):
        tmpdata = data[np.where(Adjacent[i,:]!=0)[0],:]
        if tmpdata.shape[0]>2:
            tmp_std = torch.zeros((tmpdata.shape[0],tmpdata.shape[1]))
            for j in range(0,tmpdata.shape[0]):
                tmp_std[j]=(torch.mul(data[i],tmpdata[j]))
            data1[i, :]=data[i, :] + alph * tmp_std.mean()*torch.ones(1,data.shape[1])
            data2[i, :] = data[i,:] - alph * tmp_std.mean()*torch.ones(1,data.shape[1])

    return data1,data2
def createdata2(data,k,alph):
    Similarity = calEuclidDistanceMatrix(data)
    Similarity2 = calEuclidDistanceMatrix(np.transpose(data))
    data1=torch.zeros(data.size())
    data2=torch.zeros(data.size())
    data3=torch.zeros(np.transpose(data).size())
    data4=torch.zeros(np.transpose(data).size())
    Adjacent = myKNN(Similarity, k)
    Adjacent2 = myKNN(Similarity2, k)
    for i in range(Adjacent.shape[0]):
        tmpdata = data[np.where(Adjacent[i,:]!=0)[0],:]
        tmp_std = torch.zeros(1,tmpdata.shape[1])
        for l in range(tmpdata.shape[1]):
            tmp_std[:,l]=torch.std(tmpdata[:,l])
        data1[i,:]=data[i,:] + alph * tmp_std*torch.ones(1,data.shape[1])
        data2[i, :] = data[i,:] - alph * tmp_std*torch.ones(1,data.shape[1])

    for j in range(Adjacent2.shape[0]):
        tmpdata = np.transpose(data)[np.where(Adjacent2[j,:]!=0)[0],:]
        tmp_std = torch.zeros(1,data4.shape[1])
        for l in range(data4.shape[1]):
            tmp_std[:,l]=torch.std(tmpdata[:,l])
        data3[j,:]=np.transpose(data)[j,:] + alph * tmp_std*torch.ones(1,data3.shape[1])
        data4[j, :] = np.transpose(data)[j,:] - alph * tmp_std*torch.ones(1,data4.shape[1])
    return torch.FloatTensor(data1),torch.FloatTensor(data2),torch.FloatTensor(np.transpose(data3)),torch.FloatTensor(np.transpose(data4))


#data为数据集，k为k近邻数一般设置为3~15，alph为常系数一般设置为1~3
def createdatareal(data,adj,alph):
    data1=torch.zeros(data.shape)
    data2=torch.zeros(data.shape)
    Adjacent = adj.todense()
    for i in range(Adjacent.shape[0]):
        tmpdata = data[np.where(Adjacent[i,:]!=0)[0],:]
        if tmpdata.shape[0]>1:
            tmp_std = torch.zeros(1,tmpdata.shape[1])
            for l in range(tmpdata.shape[1]):
                tmp_std[:,l]=torch.std(tmpdata[:,l])
            data1[i,:]=data[i,:] + alph * tmp_std*torch.ones(1,data.shape[1])
            data2[i, :] = data[i,:] - alph * tmp_std*torch.ones(1,data.shape[1])
        else:
            data1[i,:] = data[i,:]
            data2[i, :] = data[i,:]

    return torch.FloatTensor(data1),torch.FloatTensor(data2)

def createdatadot(data,adj,alph):
    data1=torch.zeros(data.shape)
    data2=torch.zeros(data.shape)
    Adjacent = adj.todense()
    for i in range(Adjacent.shape[0]):
        tmpdata = data[np.where(Adjacent[i,:]!=0)[0],:]
        if tmpdata.shape[0]>1:
            tmp_std = torch.zeros((tmpdata.shape[0],tmpdata.shape[1]))
            for j in range(0,tmpdata.shape[0]):
                tmp_std[j]=(torch.mul(data[i],tmpdata[j]))
            data1[i, :] = data[i,:] + alph * tmp_std.mean()*torch.ones(1, data.shape[1])
            data2[i, :] = data[i,:] - alph * tmp_std.mean()*torch.ones(1, data.shape[1])
        else:
            data1[i,:] = data[i,:]
            data2[i, :] = data[i,:]

    return torch.FloatTensor(data1),torch.FloatTensor(data2)


def Augdata(data,adj,alph):
    data1=torch.zeros(data.shape)
    data2=torch.zeros(data.shape)
    data3=torch.zeros(data.shape)
    data4=torch.zeros(data.shape)
    Adjacent = torch.zeros(adj.to_dense().shape)
    neg_Adjacent = torch.ones(adj.to_dense().shape)
    Adjacent[np.where(adj.to_dense()!=0)[0],np.where(adj.to_dense()!=0)[1]]=1
    neg_Adjacent = neg_Adjacent - Adjacent

    diag = torch.diag(Adjacent)  # 取 a 对角线元素，输出为 1*3
    a_diag = torch.diag_embed(diag)  # 由 diag 恢复为三维 3*3
    Adjacent = Adjacent - a_diag  # a 对角线置 0
    pos_aug = torch.spmm(Adjacent, data)
    neg_aug = torch.spmm(neg_Adjacent, data)

    data1=data+pos_aug
    data2=data-pos_aug
    data3=data+neg_aug
    data4=data-neg_aug


    return torch.FloatTensor(data1),torch.FloatTensor(data2),torch.FloatTensor(data3),torch.FloatTensor(data4)

def Augdata_mean(data,adj,alph):
    data1=torch.zeros(data.shape)
    data2=torch.zeros(data.shape)
    data3=torch.zeros(data.shape)
    data4=torch.zeros(data.shape)
    Adjacent = torch.FloatTensor(adj.to_dense())
    # neg_Adjacent = torch.ones(adj.todense().shpe)
    # Adjacent[np.where(adj.todense()!=0)[0],np.where(adj.todense()!=0)[1]]=1
    # neg_Adjacent = neg_Adjacent - Adjacent

    # diag = torch.diag(Adjacent)  # 取 a 对角线元素，输出为 1*3
    # a_diag = torch.diag_embed(diag)  # 由 diag 恢复为三维 3*3
    # Adjacent = Adjacent - a_diag  # a 对角线置 0
    pos_aug = torch.spmm(Adjacent, data)
    # neg_aug = torch.spmm(neg_Adjacent, data)

    data1=data+pos_aug
    data2=data-pos_aug
    # data3=data+neg_aug
    # data4=data-neg_aug


    return torch.FloatTensor(data1),torch.FloatTensor(data2)
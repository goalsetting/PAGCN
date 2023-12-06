
import os.path as osp
import torch
from utilstools.dataset import get_dataset

##这个函数为torch_geometric库数据的读取函数，以及数据类型的转换

def adjacency_matrix_to_edge_index(adjacency_matrix):
    edge_index = []
    num_nodes = adjacency_matrix.shape[0]

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i][j] != 0:
                edge_index.append([i, j])

    # Transpose the edge list for PyTorch Geometric convention
    edge_index = torch.tensor(edge_index).t().contiguous()

    return edge_index

def edge_index_to_adjacency_matrix(edge_index, num_nodes):
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))

    # Assuming edge_index is a tensor of shape (2, num_edges)
    # where each column represents an edge (source node, target node)
    for src, tgt in edge_index.t():
        adjacency_matrix[src][tgt] = 1
        adjacency_matrix[tgt][src] = 1  # Uncomment for undirected graph

    return adjacency_matrix

#12个数据集，建议学习一下torch_geometric库函数的使用定义图卷积层，这样可以不用数据预处理。FutherGroup.py有个例子
#'Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy','Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code'
path = osp.expanduser('~/datasets')
path = osp.join(path, "Amazon-Photo")
dataset = get_dataset(path, "Amazon-Photo")

data = dataset[0]

#转化为临界矩阵的格式
adj=edge_index_to_adjacency_matrix(dataset.edge_index,  dataset[0].shape[0])

labels=dataset.y

#标签率的设置参考原始的loaddata代码
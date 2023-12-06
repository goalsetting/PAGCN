from utils.load_DBLP import load_DBLPdataset
from utils.utils2 import load_data3, sparse_mx_to_torch_sparse_tensor, load_adjs
#这个函数为自己编写的读取数据整合函数
def load_data(name,lr):
    if name == 'DBLP':
        data, adj, labels, idx_train, idx_val, idx_test, idx_unlabel = load_DBLPdataset("./data/dblp.npz", lr)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        # data = normalize(data.astype(dtype=float))
        # data = sparse_mx_to_torch_sparse_tensor(data)
        return adj,data,labels, idx_train, idx_val, idx_test, idx_unlabel
    else:
        if name == 'acm':
            #  能不能设置成输出到文件，再从文件读的形式。
            adj, adj2, adj3, adj4, adj5, adj6 = load_adjs(path="./data/" + name, dataset=name)
            adj = sparse_mx_to_torch_sparse_tensor(adj)

            data, labels, idx_train, idx_val, idx_test, idx_unlabel = load_data3(path="./data/" + name + "/",
                                                                                     dataset=name,
                                                                                     labelRatio=lr)
            return adj, data, labels, idx_train, idx_val, idx_test, idx_unlabel
        else:
            adj, adj2, adj3, adj4, adj5, adj6 = load_adjs(path="./data/", dataset=name)
            adj = sparse_mx_to_torch_sparse_tensor(adj)
            data, labels, idx_train, idx_val, idx_test, idx_unlabel = load_data3(path="./data/", dataset=name,
                                                                                 labelRatio=lr)
            return adj, data, labels, idx_train, idx_val, idx_test, idx_unlabel
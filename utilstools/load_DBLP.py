import math

import numpy as np
import scipy.sparse as sp
import torch


def load_DBLPdataset(file_name,lr):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle = True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        no_class = set(z)

        idx = np.arange(len(z))

        labels = torch.LongTensor(z)
        np.random.shuffle(idx)
        idx = np.arange(len(labels))
        # 去掉离群点15个
        empetySet = []
        idx = list(set(idx).difference(empetySet))




        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        np.random.shuffle(idx)

        # print(idx)
        labelRatio=lr
        train_size = math.ceil(len(labels) * labelRatio / 100 / len(no_class))
        #print(math.ceil(len(labels) * labelRatio / 100 / len(no_class)))

        train_size = [train_size for i in range(len(no_class))]

        idx_train = []
        count = [0 for i in range(len(no_class))]
        label_each_class = train_size
        next = 0


        for i in idx:
            if count == label_each_class:
                break
            next += 1
            for j in range(len(no_class)):

                if j== labels[i] and count[j] < label_each_class[j]:
                    idx_train.append(i)
                    count[j] += 1

        idx_val = idx[next:next + 500]

        test_size = 1000
        idx_test = idx[-test_size:] if test_size else idx[next:]


        idx_unlabel = list(set(idx).difference(set(idx_train)))

        # graph = {
        #     'A': A,
        #     'X': X,
        #     'z': z
        # }

        # idx_to_node = loader.get('idx_to_node')
        # if idx_to_node:
        #     idx_to_node = idx_to_node.tolist()
        #     graph['idx_to_node'] = idx_to_node
        #
        # idx_to_attr = loader.get('idx_to_attr')
        # if idx_to_attr:
        #     idx_to_attr = idx_to_attr.tolist()
        #     graph['idx_to_attr'] = idx_to_attr
        #
        # idx_to_class = loader.get('idx_to_class')
        # if idx_to_class:
        #     idx_to_class = idx_to_class.tolist()
        #     graph['idx_to_class'] = idx_to_class

        return X,A, labels, idx_train, idx_val, idx_test,idx_unlabel

# g = load_dataset('data/DBLP.npz')
# A, X, z = g['A'], g['X'], g['z']
# 1
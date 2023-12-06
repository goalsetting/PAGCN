# 读取数据
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F

from Reddit import  SAGE
#这个函数为torch_geometric库定义卷积层，以及读取数据分割数据的函数

transform = T.ToSparseTensor()
# 这里加上了ToSparseTensor()，所以边信息是以adj_t形式存储的，如果没有这个变换，则是edge_index
dataset = Planetoid(name='Citeseer', root=r'./dataset/Cora', transform=transform)
data = dataset[0]
device = torch.device('cpu')
data.adj_t = data.adj_t.to_symmetric()




model2 = SAGE(in_channels=dataset.num_features, hidden_channels=64, out_channels=dataset.num_classes)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)




def train2():
    model2.train()

    optimizer2.zero_grad()
    out2= model2(data.x, data.adj_t)[data.train_mask]  # 前面我们提到了，SAGE是实现了edge_index和adj_t两种形式的
    loss2 = F.nll_loss(out2, data.y[data.train_mask])
    loss2.backward()
    optimizer2.step()

    return loss2.item()


@torch.no_grad()
def test2():
    model2.eval()

    out2 = model2(data.x, data.adj_t)
    y_pred2 = out2.argmax(axis=-1)

    correct2 = y_pred2 == data.y
    train_acc2 = correct2[data.train_mask].sum().float() / data.train_mask.sum()
    valid_acc2 = correct2[data.val_mask].sum().float() / data.val_mask.sum()
    test_acc2 = correct2[data.test_mask].sum().float() / data.test_mask.sum()

    return train_acc2, valid_acc2, test_acc2


# 跑10个epoch看一下模型效果
for epoch in range(70):

    loss2 = train2()
    train_acc2, valid_acc2, test_acc2 = test2()

    print(f'Epoch: {epoch:02d}, '
          f'rawLoss: {loss2:.4f}, '
          f'rawTrain_acc: {100 * train_acc2:.3f}%, '
          f'rawValid_acc: {100 * valid_acc2:.3f}% '
          f'rawTest_acc: {100 * test_acc2:.3f}%')

import random

import torch
#这个函数为噪声实验函数，鲁棒性实验还包括低标标签率，可在Main函数中调节

#随机减边，在读取数据操作时加入
def inject_adjnosie(graph,r):
    ran = random.sample(range(0, len(graph)), int(len(graph)*r))
    for i in ran:
        tmp = graph[i]
        for jj in tmp:
            graph[jj].remove(i)
        graph[i]=[]
    return graph

#随机生成边，在读取数据是操作
def inject_adjnosie2(graph,r):
    ran = random.sample(range(0, len(graph)), int(len(graph)*r))
    for i in ran:
        tmp = graph[i]
        for j in range(len(tmp)):
            graph[tmp[j]].remove(i)
        graph[i] = random.sample(range(0, len(graph)), random.randint(0,int(len(graph)*r)))
        for jj in range(len(graph[i])):
            if i not in graph[graph[i][jj]]:
               graph[graph[i][jj]].append(i)
    return graph

#随机掩盖特征节点，在读取数据后操作
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

#节点特征噪声得到注入，在读取数据后操作
def attack_feature(features, noise_level=0.1):
    # 生成符合正态分布的噪声
    noise = torch.randn(features.shape) * noise_level

    # 对节点特征应用噪声攻击
    attacked_features = features + noise
    return attacked_features


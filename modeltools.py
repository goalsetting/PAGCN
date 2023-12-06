import torch.nn as nn

from AttentionModule import Attention
# from Visualization import t_SNE, Visualization
from layertools import GraphConvolution
import torch
import torch.nn.functional as F

#PAGCN(GCNPE)的模型定义

#GCN网络模型
class GCNPE(nn.Module):
    #初始化操作
    def __init__(self, nfeat, nhid, nclass,dropout,tau: float = 0.9):
        super(GCNPE, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        self.attention = Attention(nclass)

        # self.fc1 = torch.nn.Linear(nclass, 64)
        # self.fc2 = torch.nn.Linear(64,nfeat)

        self.fc11 = torch.nn.Linear(nclass, 64)
        self.fc22 = torch.nn.Linear(64,nclass)
        self.tau=tau

    #前向传播
    def forward(self, x, datareal1, datareal2,adj,labels=[]):
        outputs=[]
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        output1 = F.log_softmax(self.gc2(x, adj))

        datareal1 = F.relu(self.gc1(datareal1, adj))
        datareal1 = F.dropout(datareal1, self.dropout, training=self.training)
        output2 = F.log_softmax(self.gc2(datareal1, adj))

        datareal2 = F.relu(self.gc1(datareal2, adj))
        datareal2 = F.dropout(datareal2, self.dropout, training=self.training)
        output3 = F.log_softmax(self.gc2(datareal2, adj))

        # output1 = self.projection(output1)
        # output2 = self.projection(output2)
        # output3 = self.projection(output3)


        # outputs.append(output1)
        outputs.append(output1)
        outputs.append(output2)
        outputs.append(output3)


        attinputs = torch.stack(outputs, dim=1)
        tmpoutput , att = self.attention(attinputs)
        #可视化
        # if labels!=[]:
        #     output = tmpoutput.cpu().detach().numpy()
        #     labels = labels.cpu().detach().numpy()
        #
        #     result = t_SNE(output, 2)
        #     Visualization(result, labels)

        finaloutput = F.log_softmax(tmpoutput, dim=1)


        return outputs,finaloutput

    # def projection(self, z: torch.Tensor) -> torch.Tensor:
    #     z = F.elu(self.fc1(z))
    #     return self.fc2(z)
    def projection_node(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc11(z))
        return self.fc22(z)

class GCNPE_mean(nn.Module):
    #初始化操作
    def __init__(self, nfeat, nhid, nclass,dropout,tau: float = 0.9):
        super(GCNPE_mean, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        # self.attention = Attention(nclass)
        #
        # self.fc1 = torch.nn.Linear(nclass, 64)
        # self.fc2 = torch.nn.Linear(64,nfeat)

        self.fc11 = torch.nn.Linear(nclass, 64)
        self.fc22 = torch.nn.Linear(64,nclass)
        self.tau=tau

    #前向传播
    def forward(self, x, datareal1, datareal2,adj):
        outputs=[]
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        output1 = F.log_softmax(self.gc2(x, adj))

        datareal1 = F.relu(self.gc1(datareal1, adj))
        datareal1 = F.dropout(datareal1, self.dropout, training=self.training)
        output2 = F.log_softmax(self.gc2(datareal1, adj))

        datareal2 = F.relu(self.gc1(datareal2, adj))
        datareal2 = F.dropout(datareal2, self.dropout, training=self.training)
        output3 = F.log_softmax(self.gc2(datareal2, adj))

        # output1 = self.projection(output1)
        # output2 = self.projection(output2)
        # output3 = self.projection(output3)


        # outputs.append(output1)
        outputs.append(output1)
        outputs.append(output2)
        outputs.append(output3)


        # outputs.append(negoutput1)
        # outputs.append(negoutput2)
        # attinputs = torch.stack(outputs, dim=1)
        # tmpoutput , att = self.attention(attinputs)
        # finaloutput = F.log_softmax(tmpoutput, dim=1)
        finaloutput=torch.mean(torch.stack(outputs[0:3]), dim=0, keepdim=True)



        return outputs,finaloutput

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    def projection_node(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc11(z))
        return self.fc22(z)

#GCN网络模型
class GCN(nn.Module):
    #初始化操作
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    #前向传播
    def forward(self, x, adj, labels=[]):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        #可视化
        # if labels!=[]:
        #     output = x.cpu().detach().numpy()
        #     labels = labels.cpu().detach().numpy()
        #
        #     result = t_SNE(output, 2)
        #     Visualization(result, labels)
        return F.log_softmax(x, dim=1)
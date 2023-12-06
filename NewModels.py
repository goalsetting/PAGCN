import torch
from torch import optim
import torch.nn.functional as F

from modeltools import GCN, GCNPE
from utilstools.utils2 import accuracy
#PAGCN(GCNPE)的模型封装
class TrainModel:
    def __init__(self, features, args, labels, name='GCN',device=torch.device("cpu")):
        if name == 'GCN':
            model = GCN(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=int(labels.max().item()) + 1,
                        dropout=args.dropout).to(device=device)
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
        if name == 'GCNPE':
            model = GCNPE(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=int(labels.max().item()) + 1,
                        dropout=args.dropout).to(device=device)
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
        if name == 'GCNPE_mean':
            model = GCNPE(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=int(labels.max().item()) + 1,
                        dropout=args.dropout).to(device=device)
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
        self.model = model
        self.optimizer = optimizer
        self.features = features
        self.args = args
        self.labels = labels



    def GCNPEtrain(self, data, datarea1,datareal2,Adj, idx_train,idx_val,idx_unlabel,lamd=0.5):
        # input = (data, datarea1,datareal2,Adj,)
        # flops, params = profile(self.model,input)
        #
        # print(f"FLOPs: {flops}")
        # print(f"Computation (in Million): {flops / 1e6}")
        self.model.train()
        # 初始化梯度函数
        self.optimizer.zero_grad()

        outputs,output=self.model(data, datarea1,datareal2,Adj)


        loss_xy = F.nll_loss(output[idx_train], self.labels[idx_train].long())

        outputrealnode = self.model.projection_node(outputs[0][idx_unlabel])
        output2realnode = self.model.projection_node(outputs[1][idx_unlabel])
        output3realnode = self.model.projection_node(outputs[2][idx_unlabel])
        # 多向性损失
        loss_ncl2 = F.mse_loss(outputrealnode, output2realnode)
        # 多向性损失
        loss_ncl3 = F.mse_loss(outputrealnode, output3realnode)
        # 多向性损失
        loss_ncl4 = F.mse_loss(output2realnode, output3realnode)
        loss_constract = (loss_ncl2+loss_ncl3)/loss_ncl4

        loss_train = (1-lamd)*loss_xy+lamd*loss_constract
        # loss_train=loss_xy

        # 梯度下降
        loss_train.backward()
        # 参数优化
        self.optimizer.step()

        #   for val
        # 关闭参数调整过程，单纯的展示目前训练结果
        self.model.eval()
        _,output= self.model(data,datarea1,datareal2,Adj)

        # print(len(output[0]))
        loss_val = F.nll_loss(output[idx_val], self.labels[idx_val])

    def GCNPEtrain_loss(self, data, datarea1,datareal2,Adj, idx_train,idx_val,idx_unlabel,lamd=0.5):
        self.model.train()
        # 初始化梯度函数
        self.optimizer.zero_grad()

        outputs,output=self.model(data, datarea1,datareal2,Adj)

        loss_xy = F.nll_loss(output[idx_train], self.labels[idx_train].long())

        # outputrealnode = self.model.projection_node(outputs[0][idx_unlabel])
        # output2realnode = self.model.projection_node(outputs[1][idx_unlabel])
        # output3realnode = self.model.projection_node(outputs[2][idx_unlabel])
        # # 多向性损失
        # loss_ncl2 = F.mse_loss(outputrealnode, output2realnode)
        # # 多向性损失
        # loss_ncl3 = F.mse_loss(outputrealnode, output3realnode)
        # # 多向性损失
        # loss_ncl4 = F.mse_loss(output2realnode, output3realnode)
        # loss_constract = (loss_ncl2+loss_ncl3)/loss_ncl4

        loss_train = loss_xy
        # loss_train=loss_xy

        # 梯度下降
        loss_train.backward()
        # 参数优化
        self.optimizer.step()

        #   for val
        # 关闭参数调整过程，单纯的展示目前训练结果
        self.model.eval()
        _,output= self.model(data,datarea1,datareal2,Adj)
        # print(len(output[0]))
        loss_val = F.nll_loss(output[idx_val], self.labels[idx_val])

    def origintrain(self, data, Adj, idx_train,idx_val):
        self.model.train()
        # 初始化梯度函数
        self.optimizer.zero_grad()

        output = self.model(data, Adj)


        loss_xy = F.nll_loss(output[idx_train], self.labels[idx_train])

        loss_train = loss_xy
        # 梯度下降
        loss_train.backward()
        # 参数优化
        self.optimizer.step()

        #   for val
        # 关闭参数调整过程，单纯的展示目前训练结果
        self.model.eval()
        output = self.model(data, Adj)
        # print(len(output[0]))
        loss_val = F.nll_loss(output[idx_val], self.labels[idx_val])


    def test(self,adj,adjtensor,idx_test,data1,data2,data3,data4,savename):
        self.model.eval()
        output = self.model(data1,data2,data3,data4,adj, adjtensor)
        areout = output[1]
        loss_test = F.nll_loss(areout[idx_test], self.labels[idx_test])
        acc_test = accuracy(areout[idx_test], self.labels[idx_test])
        print(savename+"Test set results:", "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return areout

    def test2(self,adjtensor,idx_test,data1,data2,data3,data4,savename):
        self.model.eval()
        output = self.model(data1,data2,data3,data4, adjtensor)
        areout = output[1]
        loss_test = F.nll_loss(areout[idx_test], self.labels[idx_test])
        acc_test = accuracy(areout[idx_test], self.labels[idx_test])

        print(savename+"Test set results:", "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return areout

    def testorigin(self,adjtensor,idx_test,K,lamd,name,savename):
        self.model.eval()
        output = self.model(self.features, adjtensor,self.labels)
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = accuracy(output[idx_test], self.labels[idx_test])
        preds = output[idx_test].max(1)[1].type_as(self.labels[idx_test])
        print(savename+"Test set results:", "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return preds,acc_test


    def testgcnpe(self,data1real,data2real,adjtensor,idx_test,K,lamd,name,savename):
        self.model.eval()
        _,output= self.model(self.features,data1real,data2real,adjtensor, self.labels)
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = accuracy(output[idx_test], self.labels[idx_test])
        preds = output[idx_test].max(1)[1].type_as(self.labels[idx_test])
        print(savename+"Test set results:", "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return preds,acc_test
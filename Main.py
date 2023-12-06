import argparse

import numpy as np
import torch

from NewModels import TrainModel
from utilstools.LoadData import load_data
#PAGCN的主函数
def createdataaug(data,adj,alph,devicestr):
    data=data.to(torch.device(devicestr))
    data1=torch.zeros(data.size()).to(torch.device(devicestr))
    data2=torch.zeros(data.size()).to(torch.device(devicestr))
    Adjacent = adj

    for i in range(Adjacent.shape[0]):
        tmpdata = data[torch.where(Adjacent[i,:]!=0)[0],:]
        if tmpdata.shape[0]>2:
            tmp_std = torch.zeros(1,tmpdata.shape[1]).to(torch.device(devicestr))
            for l in range(tmpdata.shape[1]):
                tmp_std[:,l]=torch.std(tmpdata[:,l])
            data1[i, :]=data[i, :] + alph * tmp_std*torch.ones(1,data.shape[1]).to(torch.device(devicestr))
            data2[i, :] = data[i,:] - alph * tmp_std*torch.ones(1,data.shape[1]).to(torch.device(devicestr))

    return data1,data2

def doGCNPE(se,lr,features, datareal1, datareal2,labels, idx_train, idx_val, idx_test, idx_unlabel,validate,hiddenNum,name,adj,devicestr):

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=se, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=hiddenNum,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Tolerance for early stopping (# of epochs).')


    parser.add_argument('--labelRatio', type=float, default=lr,
                        help='label sample ratio(1%) in all samples.')

    parser.add_argument('--degree', type=int, default=2,
                        help='the degree')
    # GFNN参数
    parser.add_argument('--batch_size2', type=int, default=10)

    # checkpt_file = '.mod_cora.ckpt'
    # args=[]
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device=torch.device(devicestr)


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        features=features.to(device)
        datareal2 = datareal2.to(device)
        adj=adj.to(device)
        datareal1=datareal1.to(device)
        labels = labels.to(device)
        idx_train=idx_train.to(device)
        idx_val=idx_val.to(device)
        idx_test=idx_test.to(device)

    print(" adj\t" + str(lr) + " %label\t" + str(iter) + " iter\t" )

    ACCGCNPE=np.zeros([1,len(range(5,100,5))])
    index=0
    for betw in range(5, 100, 5):
        betwT = betw / 100
        print(" adj\t" + str(lr) + " %label\t" + str(iter) + " iter\t" + str(betwT) + " lamd\t")
        modelGCNPE = TrainModel(features=features,
                                args=args,
                                labels=labels
                                , name='GCNPE',device=device)

        for epoch in range(args.epochs):
            modelGCNPE.GCNPEtrain(features,datareal1,datareal2, adj, idx_train, idx_val, idx_unlabel,betwT)
        areoutpe = modelGCNPE.testgcnpe(datareal1,datareal2,adj, idx_test, lr, betwT, name, "GCNPE")
        ACCGCNPE[0,index]=areoutpe[1]
        index=index+1

        modelGCNPE.optimizer.zero_grad()
    print("acc:",ACCGCNPE.max())
    return ACCGCNPE.max()


if __name__ == '__main__':
#DBLP Acm texas cornell coraml chameleon cora citeseer pubmed可以直接读取
#Reddit PPI 亚马逊等社交网络数据集需要torch_geometric库读取并转化，详细参考FutherGroup reddit_graphsage_supervised的读取方式
    datasetStr = "acm"
    devicestr='cuda' #是否启用GPU
    name = datasetStr
    validate = False

    hiddenNum = 128

    name = datasetStr
    validate = False
    #(14,21,28,56)


    # datareal1 = torch.FloatTensor(np.load('./data/data1real' + name + '.npy', allow_pickle=True))
    # datareal2 = torch.FloatTensor(np.load('./data/data2real' + name + '.npy', allow_pickle=True))



    # LabelRatioList = [4]
    LabelRatioList = [4]
    adj, features, _, _, _, _, _ = load_data(name, lr=10)
    #数据增强函数
    datareal1, datareal2 = createdataaug(features.to(torch.device(devicestr)), adj.to_dense().to(torch.device(devicestr)), 1,devicestr)
    #建议对结果保存，下次不用重复跑
    np.save('./data/data1real' + name + '.npy', datareal1.cpu().numpy())
    np.save('./data/data2real' + name + '.npy', datareal2.cpu().numpy())
    for lr in LabelRatioList:
        ACCGCN=np.zeros([1,1])
        ACCGCNPE=np.zeros([1,1])
        ACCSGC = np.zeros([1, 1])
        ACCGFNN = np.zeros([1, 1])
        for iter in range(0, 1, 1):
            se  = 39788

            adj, features, labels, idx_train, idx_val, idx_test, idx_unlabel = load_data(name, lr)

            idx_train, idx_val, idx_test, idx_unlabel = list(
                map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test, idx_unlabel]))



            ACCGCNPE[0,iter]=doGCNPE(se,lr,features, datareal1, datareal2,labels.long(), idx_train.long(), idx_val.long(), idx_test.long(), idx_unlabel.long(), validate, hiddenNum,name,adj,devicestr)

        print(str(ACCGCNPE.mean()) + " %gcnpe\t ")
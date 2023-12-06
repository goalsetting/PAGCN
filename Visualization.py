# t-SNE 降维
from matplotlib import pyplot as plt
from sklearn import manifold
#可视化实验函数

def t_SNE(output, dimention):
    # output:待降维的数据
    # dimention：降低到的维度
    tsne = manifold.TSNE(n_components=dimention, init='pca', random_state=0)
    result = tsne.fit_transform(output)
    return result

# Visualization with visdom
def Visualization(result, labels):
    # #可视化
    color = ['#F0F8FF', 'green', 'b', 'r', '#7FFFD4', '#FFC0CB', '#00022e']
    for index, item in enumerate(result):
        plt.scatter(item[0], item[1], c=color[labels[index]])
        # 140,28,21,14
    plt.savefig( 'Cora140GCN.eps',dpi= 600, format = 'eps')
    plt.show()



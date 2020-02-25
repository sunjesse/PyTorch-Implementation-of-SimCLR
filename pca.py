from __future__ import print_function
import time
import numpy as np
import pandas as pd
import argparse
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
from models.models import ResNet18, ConvLSTM, ClassifierModule, ClassifierModuleDense
from utils import utils
from lib import radam
from loss import SimLoss

def dataloader(args):
    if args.dataset.lower() == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor()])
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=2)


        return testloader

def test(net, testloader, args):
    net.eval()
    a = torch.empty(testloader.__len__(), 128)
    l = torch.empty(testloader.__len__())
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            outputs = net(inputs)
            a[i] = outputs
            l[i] = labels

    print(a.shape)
    print(l.shape)
    
    feat_cols = [ 'pixel'+str(i) for i in range(a.shape[1]) ]
    df = pd.DataFrame(a,columns=feat_cols)
    df['y'] = l
    df['label'] = df['y'].apply(lambda i: str(i))

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1]
    df['pca-three'] = pca_result[:,2]
    
    rndperm = np.random.permutation(df.shape[0])

    plt.figure(figsize=(16,10))
    p = sns.scatterplot(
            x="pca-one", y="pca-two",
            hue="y",
            palette=sns.color_palette("hls", 10),
            data=df.loc[rndperm,:],
            legend="full",
            alpha=0.3
        )
    p.get_figure().savefig("./pca.png")

    

def checkpoint(net, args, epoch_num):
    print('Saving checkpoints...')

    suffix_latest = 'epoch_{}.pth'.format(epoch_num)
    dict_net = net.state_dict()
    torch.save(dict_net,
               '{}/resnet_{}'.format(args.ckpt, suffix_latest))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # optimization related arguments
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--tau', default=0.1, type=float)
    parser.add_argument('--ckpt', default="/home/rexma/Desktop/JesseSun/simclr")
    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    args.num_class = 10 if args.dataset.lower() == 'cifar10' else 1000

    testloader = dataloader(args)

    net = ResNet18(mode='test').cuda()
    
    net.load_state_dict(
                    torch.load(args.ckpt, map_location=lambda storage, loc: storage), strict=False)
    print("Loaded pretrained weights.")

    test(net, testloader, args)


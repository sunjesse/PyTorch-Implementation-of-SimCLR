import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
from data.ucf101 import UCF101
from models.models import ResNet18, ConvLSTM, ClassifierModule, ClassifierModuleDense
from utils import utils 
from lib import radam
from loss import SimLoss

s=1
color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
data_augment = transforms.Compose([transforms.ToPILImage(),
                                   transforms.RandomResizedCrop(32),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomApply([color_jitter], p=0.8),
                                   transforms.RandomGrayscale(p=0.2),
                                   utils.GaussianBlur(),
                                   transforms.ToTensor()])

def dataloader(args):
    if args.dataset.lower() == 'cifar10':
        transform = transforms.Compose(
	    [transforms.ToTensor()])
	    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
					        download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
					        shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
		      			    download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
					     shuffle=False, num_workers=2)


        return trainloader, testloader

def optimizer(net, args):
    assert args.optimizer.lower() in ["sgd", "adam", "radam"], "Invalid Optimizer"

    if args.optimizer.lower() == "sgd":
	       return optim.SGD(net.parameters(), lr=args.lr, momentum=args.beta1, nesterov=args.nesterov)
    elif args.optimizer.lower() == "adam":
	       return optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer.lower() == "radam":
            return radam.RAdam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

def test(net, epoch, criterion, testloader, args):
    net.eval()
    with torch.no_grad():
        correct = 0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            pred = F.softmax(outputs, 1)
            _, pred = torch.max(pred, 1)
            correct += torch.sum(pred==labels)
        print("Test set accuracy: " + str(float(correct)/ float(testloader.__len__())))

def train(net, epoch, criterion, optimizer, trainloader, args):
    loss_meter = utils.AverageMeter()
    net.train()

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        loss_meter.update(loss.item())
        optimizer.step()

        #running_loss += loss.item()
        if i % 1 == 0 and i > 0:
            print('[Epoch %02d, Minibatch %05d] Loss: %.5f' %
			(epoch, i, loss_meter.average()))
            #running_loss = 0.0

def SimCLR(net, epoch, criterion, optimizer, trainloader, args):
    loss_meter = utils.AverageMeter()
    net.train()
    
    for i, data in enumerate(trainloader, 0):
        b, _ = data
        optimizer.zero_grad()
        x_1 = torch.zeros_like(b).cuda()
        x_2 = torch.zeros_like(b).cuda()

        for idx, x in enumerate(b):
            x_1[idx] = data_augment(x)
            x_2[idx] = data_augment(x)
        #b = b.cuda()
        out_1 = net(x_1)
        out_2 = net(x_2)
        
        loss = criterion(torch.cat([out_1, out_2], dim=0))
        loss.backward()
        loss_meter.update(loss.item())
        optimizer.step()

        if i % 100 == 0 and i > 0:
            print('[Epoch %02d, Minibatch %05d] Loss: %.5f' %
                        (epoch, i, loss_meter.average()))

def checkpoint(net, args, epoch_num):
    print('Saving checkpoints...')
    
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)
    dict_net = net.state_dict()
    torch.save(dict_net,
               '{}/resnet_{}'.format(args.ckpt, suffix_latest))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # optimization related arguments
    parser.add_argument('--batch_size', default=4, type=int,
                        help='input batch size')
    parser.add_argument('--epoch', default=100, type=int,
                        help='epochs to train for')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--lr', default=0.001, type=float, help='LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--nesterov', default=False)
    parser.add_argument('--tau', default=1, type=float)
    parser.add_argument('--ckpt', default="~/Desktop/JesseSun/simclr")
    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))
    
    args.num_class = 10 if args.dataset.lower() == 'cifar10' else 1000

    trainloader, testloader = dataloader(args)

    net = ResNet18().cuda()
    criterion = SimLoss(tau=args.tau).cuda()
    optimizer = optimizer(net, args)
    for epoch in range(1, args.epoch+1):
        SimCLR(net, epoch, criterion, optimizer, trainloader, args)
        #test(net, epoch, criterion, testloader, args)
        #if epoch%5==0:
        #    checkpoint(net, args, epoch)
    
    print("Training completed!")

#coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from data_test_endtoend import Data10##########
#from s2d import Scattering2D##############

import os
import time
import numpy as np

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'######在网络文件中指定单gpu,尺寸更统一


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
'''
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
'''

# DataLoaders################

num_workers = 2


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
root='/home/code/0Demo_data'
trainloader = torch.utils.data.DataLoader(
    Data10(root, train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=128, shuffle=True, num_workers=num_workers)

testloader = torch.utils.data.DataLoader(
    Data10(root, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=128, shuffle=False, num_workers=num_workers)
##################################

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

#net = ResNet18()
# net = PreActResNet18()
#net = GoogLeNet()
#net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
#net = SENet18()
# net = ShuffleNetV2(1)
#net = EfficientNetB0()
net = VGG()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()


#scattering = Scattering2D(J=2, shape=(32, 32))##############################
#scattering = scattering.cuda()################






# Training
def train(epoch, optimizer):#########
    print('Epoch {}/{}'.format(epoch + 1, 130))##############200
    print('-' * 10)
    start_time = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0


    tmpp1 = np.zeros((10 , 16, 32, 32))###########
    tmpp2 = np.zeros((10 , 81, 32, 32))###########

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)#(128, 3, 32, 32)
        #scatter_feature = scattering(inputs) ##########128,3,81,8,8########

        optimizer.zero_grad()
        outputs = net(inputs)###########################
        #print (scatter_feature)
        #tmp1 = tmp.cpu().detach().numpy()################
        #tmpp1 = tmp1[0:10,:,:,:]###############
        #tmp2 = scatter_feature.cpu().detach().numpy()################
        #tmpp2 = tmp2[0:10,:,:,:]################

        #print (inputs.shape,outputs.shape)
        loss = criterion(outputs, targets)
        #print ("losssssssssssssssssssssssssssssssssssssssssssssssssssssssss",loss)
        #print ("tttttttttttttttttttttttttttttttttttttttttttttt",targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    end_time = time.time()



    #tmpp1 = tmpp1[:,:,:,:]#############################
    #print (tmpp1.shape)
    #tmpp2 = tmpp2[:,:,:,:]#############################
    #np.save("./" + str(epoch) + "_tmpp1.npy",tmpp1)###########
    #np.save("./" + str(epoch) + "_tmpp2.npy",tmpp2)########### 


    print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, end_time-start_time))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            scatter_feature = scattering(inputs) ##########128,3,81,8,8########
   
            outputs = net(inputs)#############################
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net.state_dict(), './checkpoint/444.pth')
        best_acc = acc




#main function
time_all_start = time.time()

lr = 0.001###0.001
for epoch in range(start_epoch, start_epoch+130):###########200
    if epoch%20==0:####
        optimizer = optim.Adam(net.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)############# eps=1e-08

        lr*=0.2

    train(epoch, optimizer)#####
    test(epoch)

#print(best_acc)
#------------------------------------------------------------------
# Loading weight files to the model and testing them.
net_test = VGG()#############
net_test = net_test.to(device)
net_test = torch.nn.DataParallel(net_test)

net_test.load_state_dict(torch.load('./checkpoint/444.pth'))

net_test.eval()

test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net_test(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total

time_all_end = time.time()
print('totally cost',time_all_end-time_all_start)

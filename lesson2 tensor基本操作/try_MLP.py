import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets
import numpy as np
import cv2
from torch.autograd import Variable

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        # 标准正态分布
        self.w1 = torch.randn(200, 784, requires_grad=True)
        self.b1 = torch.zeros(200, requires_grad=True)
        self.w2 = torch.randn(100, 200, requires_grad=True)
        self.b2 = torch.zeros(100, requires_grad=True)
        self.w3 = torch.randn(num_classes, 100, requires_grad=True)
        self.b3 = torch.zeros(num_classes, requires_grad=True)

        torch.nn.init.kaiming_normal_(self.w1)
        torch.nn.init.kaiming_normal_(self.w2)
        torch.nn.init.kaiming_normal_(self.w3)
    def forward(self, x):
        x = x @ self.w1.t() + self.b1
        x = F.relu(x)
        x = x @ self.w2.t() + self.b2
        x = F.relu(x)
        x = x @ self.w3.t() + self.b3
        x = F.relu(x)
        return x

# class MLP(nn.Module):
#     def __init__(self, num_classes=10):
#         super(MLP, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(784, 200),
#             nn.ReLU(inplace=True),
#             nn.Linear(200, 200),
#             nn.ReLU(inplace=True),
#             nn.Linear(200, 20),
#             nn.ReLU(inplace=True),
#             nn.Linear(20, num_classes),
#             nn.ReLU(inplace=True)
#
#         )
#
#     def forward(self, x):
#         x = self.model(x)
#         return x
# 超参数
epoches = 25
learning_rate = 1e-2
batch_size = 200

# 导入网络
model = MLP(10)

# 损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([model.w1, model.w2, model.w3, model.b1, model.b2, model.b3], lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# load dataset
if True:
    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307, ), (0.3081, ))
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307, ), (0.3081, ))
                       ])),
        batch_size=batch_size, shuffle=True)

train_loss_list = []
accurancy_list = []

def test():
    test_loss = 0
    correct = 0

    for x, label in test_loader:
        x = x.view(-1, 784)

        out = model.forward(x)
        loss = criterion(out, label)

        pred = out.data.max(1)[1]
        correct += pred.eq(label.data).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    accurancy_list.append(correct/60000*1.0)

def train(epoches):
    for epoch in range(epoches):
        for batch_index, (x, label) in enumerate(train_loader):
            x = x.view(-1, 784)

            out = model.forward(x)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_index * len(x), len(train_loader.dataset),
                           100. * batch_index / len(train_loader), loss.item()))
                train_loss_list.append(loss.item())

        test()





train(epoches)
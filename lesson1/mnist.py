import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets
from matplotlib import pyplot as plt

# solve a problem of accurancy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# set param
batch_size = 200
learning_rate = 0.1
epochs = 10

# init global list
train_loss_list = []
accurancy_list = []

# load dataset
train_loader = DataLoader(
    datasets.MNIST('D:/data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,),(0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

test_loader = DataLoader(
    datasets.MNIST('D:/data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,),(0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

# build a multi-layer perception model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 10),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        x = self.model(x)
        return x


net = MLP()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss()

def train(epoch):
    for batch_index, (data, target) in enumerate(train_loader):
        data = data.view(-1, 784)
        data, target = data, target

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_index * len(data), len(train_loader.dataset),
                       100. * batch_index / len(train_loader), loss.item()))
            train_loss_list.append(loss.item())

def test():
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data = data.view(-1, 784)
        data, target = data, target

        logits = net(data)
        test_loss += criteon(logits, target)

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    accurancy_list.append(correct/60000*1.0)

def data_visiable(ls, name):
    plt.plot(ls)
    plt.title(name)
    plt.show()


if __name__ == "__main__":
    for epoch in range(epochs):
        train(epoch)
        test()

    data_visiable(train_loss_list, "train_loss")

    data_visiable(accurancy_list, "accurancy")

    torch.save(net, "mnist_model.pth")


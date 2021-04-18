import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import cv2
from torch.autograd import Variable
import PIL.Image as Image


class CNN(nn.Module):
    def __init__(self, input_size=224, out_size=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out_1 = nn.Linear(int(input_size*input_size), 128)
        self.out_2 = nn.Linear(128, out_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.out_1(x)
        output = self.out_2(x)

        return output

    def visualization(self, x):
        x = self.conv1(x)
        x_1 = x     # channel = 3
        x = self.conv2(x)
        x_2 = x     # channel = 16
        x = self.conv3(x)
        x_3 = x     # channel = 32

        return x_1, x_2, x_3

epoches = 30
learning_rate = 1e-3
batch_size = 4
input_size = 224
output_size = 2

model = CNN(input_size=input_size, out_size=output_size)

dataset = ImageFolder('data/dogcat_3/')
normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
transform  = T.Compose([
         T.RandomResizedCrop(input_size),
         T.RandomHorizontalFlip(),
         T.ToTensor(),
         normalize,
])

dataset = ImageFolder('data/dogcat_2/', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

def train():
    global model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epoches):
        for batch_datas, batch_labels in dataloader:
            x = batch_datas
            label = batch_labels

            out = model(x)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch)

def test():
    global model
    model.eval()
    img_read = cv2.imread('./test_img/test_img.jpg')
    img_gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (224, 224), interpolation=cv2.INTER_CUBIC)
    print(img_gray.shape)
    cv2.imwrite('./test_img/test_img_0.png', img_gray)
    img_gray.resize((1, 224, 224))
    img = np.concatenate([img_gray, img_gray, img_gray])
    img = img.reshape((1, 3, 224, 224))
    img = img.astype(np.float64) / 127.5 - 1
    torch_img = torch.from_numpy(img)
    torch_img = torch_img.float()
    torch_img = Variable(torch_img)

    x_1, x_2, x_3 = model.visualization(torch_img)

    numpy_x_1 = x_1.detach().numpy()
    numpy_x_1 = numpy_x_1.reshape((16, 112, 112))
    numpy_1 = numpy_x_1[0]
    for num in range(1, 16):
        numpy_1 += numpy_x_1[num]
    numpy_1 = numpy_1 / 16.
    numpy_1 = (numpy_1 + 1) * 127.5
    numpy_1 = numpy_1.astype(np.int)
    cv2.imwrite('./test_img/test_img_1.png', numpy_1)

    numpy_x_2 = x_2.detach().numpy()
    numpy_x_2 = numpy_x_2.reshape((32, 56, 56))
    numpy_2 = numpy_x_2[0]
    for num in range(1, 32):
        numpy_2 += numpy_x_2[num]
    numpy_2 = numpy_2 / 32.
    numpy_2 = (numpy_2 + 1) * 127.5
    numpy_2 = numpy_2.astype(np.int)
    cv2.imwrite('./test_img/test_img_2.png', numpy_2)

    numpy_x_3 = x_3.detach().numpy()
    numpy_x_3 = numpy_x_3.reshape((64, 28, 28))
    numpy_3 = numpy_x_3[0]
    for num in range(1, 64):
        numpy_3 += numpy_x_3[num]
    numpy_3 = numpy_3 / 64.
    numpy_3 = (numpy_3 + 1) * 127.5
    numpy_3 = numpy_3.astype(np.int)
    cv2.imwrite('./test_img/test_img_3.png', numpy_3)

train()
test()



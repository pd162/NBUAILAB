from torch.utils import data
import os
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader

work_path = 'D:/cifar/cifar100/train/'
filename = []

for index in os.listdir(work_path):
    filename.append(index)


transform = T.Compose([
    T.Resize(32),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    T.CenterCrop(30),  # 从图片中间切出224*224的图片
    T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1, 1]，规定均值和标准差
])

class cifar100(data.Dataset):

    def __init__(self, root, transforms=None):
        self.imgs = []
        for file_idx in os.listdir(root):
            for img_idx in os.listdir(root + '/' + file_idx):
                self.imgs.append(root + '/' + file_idx + '/' + img_idx)
        self.transforms = transforms
        # print(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        temp = img_path.split('/')[-2]
        label = filename.index(temp)
        data = Image.open(img_path)
        # transform方法很规则
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':

    dataset = cifar100('D:/cifar/cifar100/test', transforms=transform)
    img, label = dataset[0]

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2, drop_last=False)

    for batch_datas, batch_labels in dataloader:
        print(batch_datas.size(), batch_labels)
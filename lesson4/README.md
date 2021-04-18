# README



人工智能实验室lesson4 dataset实验

理论：详见PPT

代码详解

```python
from torch.utils import data
import os
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader

work_path = '.cifar/cifar100/train/' #注意修改路径
filename = []
#将label名与数字建立对应关系
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
        self.imgs = [] #包含所有图片文件路径
        for file_idx in os.listdir(root):
            for img_idx in os.listdir(root + '/' + file_idx):
                self.imgs.append(root + '/' + file_idx + '/' + img_idx)
        self.transforms = transforms
        # print(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        temp = img_path.split('/')[-2] # 取倒数第二级目录作为标签
        label = filename.index(temp) # 取出下标作为label
        data = Image.open(img_path)
        # transform方法很规则
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':

    dataset = cifar100(work_path, transforms=transform)

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2, drop_last=False)

    for batch_datas, batch_labels in dataloader:
        print(batch_datas.size(), batch_labels)
```



ps

很多同学可能不是很理解我们这节课到底在干什么，我希望能用通俗易懂的语言给大家解释一下：

我们训练的数据，有txt，excel文本文档格式，也有图片视频音频格式，它们的类别都是不同的，那么我们要想用同一种网络训练不同类别的数据，就必然会将数据转成我们的深度学习框架认识的数据类型，这个类型就是`tensor`  ，所以我们需要将数据转成tensor，然后放到自带的dataset和dataloader容器中进行图像增强等数据预处理。



如有问题欢迎在群里交流或者私戳我的qq！

pd162整理 2021/4/18
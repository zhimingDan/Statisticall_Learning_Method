# VGG表示使用重复元素的神经网络
# VGG就是使用数个相同的以3 * 3 ,填充为1的卷积核的卷积层，然后接上一个池化核为2 * 2 ，步长为2 的最大池化层
# 前面的数个3 * 3 的卷积层能够保证输入的高和宽不发生变化，后面的池化层能够使得高和宽减少一半

import torch
import d2lzh as d2l
import torch.nn as nn
from torch import optim


# VGG块的实现：
def vgg_block(num_convs, in_channels, out_channels):
    block = []
    for i in range(num_convs):
        if i == 0:
            block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
        else:
            block.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1))
        # 每个卷积层后面就一定要添加一个激活函数
        block.append(nn.ReLU())
    block.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*block)


# 构造一个VGG网络，这个网络里面中前两块使用单层卷积层，后面三块使用双层卷积层
conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# 经过5个VGG层后，输出数据的高度会减半5次
fc_features = 512 * 7 * 7
fc_hidden_units = 4096


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


# 实现VGG-11
def vgg(conv_arch, fc_features, fc_hidden_units):
    net = nn.Sequential()
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 一个类，只要继承了nn.Module就可以调用add_module方法以及_module方法
        # 因为vgg_block方式输出的nn.Sequential也是继承nn.Module，因此可以直接添加到nn.Sequential中
        net.add_module(str(i + 1), vgg_block(num_convs, in_channels, out_channels))

    # 定义全连接层部分：
    net.add_module(
        'fc', nn.Sequential(
            # 这里必须定义一个类，来改变从卷积块中输出的数据的形状
            FlattenLayer(),
            nn.Linear(fc_features, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, 10)
        )
    )
    return net


def evaluation(data_iter, net):
    acc, n = 0, 0
    for data, label in data_iter:
        data = torch.from_numpy(data.asnumpy())
        label = torch.from_numpy(label.asnumpy())
        data, label = data.to(list(net.parameters())[0].device), label.to(list(net.parameters())[0].device)
        # 进入评估模式
        net.eval()
        acc += (net(data).argmax(dim=1) == label).cpu().float().sum()
        n += label.shape[0]
    return acc / n


# 这里提前对模型的结构进行观察，定义一个高宽为224,224的数据(这个数据和样本中的数据的高宽是一样的), 然后观察每个模块的输出
# x = torch.rand(1, 1, 224, 224)
# net = vgg(conv_arch, fc_features, fc_hidden_units)
#
# for name, block in net.named_children():
#     x = block(x)
#     print('name:', name, 'output shape:', x.shape)

ratio = 4
# 通过除以相同的比例来降低模型的复杂度
small_arch = (
    (1, 1, 64 // ratio), (1, 64, 128 // ratio), (2, 128, 256 // ratio), (2, 256, 512 // ratio), (2, 512, 512 // ratio)
)

# 构建一个神经网络
net = vgg(small_arch, fc_features, fc_hidden_units).cuda(0)

# 获取数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=256)
# 定义损失函数
cross_loss = nn.CrossEntropyLoss()
device = list(net.parameters())[0].device

# 定义优化函数：
optimizer = optim.Adam(params=net.parameters(), lr=0.001)
for epoch in range(5):
    for x, y in train_iter:
        x = (torch.from_numpy(x.asnumpy())).to(device)
        y = (torch.from_numpy(y.asnumpy())).to(device)
        loss = cross_loss(net(x), y.long())
        # 进行梯度清零
        optimizer.zero_grad()
        # 进行反向传播
        loss.backward()
        # 进行梯度下降
        optimizer.step()
    print('epoch:{0}\t\t train_loss:{1}\t\t accuracy:{2}'.format(
        epoch + 1,
        loss,
        evaluation(test_iter, net)
    ))

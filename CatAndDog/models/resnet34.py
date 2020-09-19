# 就一个ResNet34模型

from .basic_module import BasicModule
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    '''
    实现module: Residual Block 残差块
    '''
    def __init__(self,inchannel,outchannel,stride = 1,shortcut = None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size = 3,stride= stride,padding=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        x = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ResNet34(BasicModule):
    '''
    主要实现module:ResNet34
    layer,layer又包含多个Residual Block
    用module 来实现Residual Block,用 _make_layer函数实现layer
    '''
    def __init__(self,num_classes = 2):
        super(ResNet34, self).__init__()
        self.model_name = 'resnet34'

        # 前基层：图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        # 重复layer,分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(64,128,3)
        self.layer2 = self._make_layer(128,256,4,stride=2)
        self.layer3 = self._make_layer(256,512,6,stride=2)
        self.layer4 = self._make_layer(512,512,3,stride=2)

        # 最后全连接层，二分类问题线性分类一样的
        self.fc = nn.Linear(512,num_classes)
        # self.softmax = nn.Softmax()

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        '''
        :param inchannel:  输入通道
        :param outchannel:  输出通道
        :param block_num:  数量
        :param stride:  步长
        :return: 迭代器序列
        '''
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))

        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x,7)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

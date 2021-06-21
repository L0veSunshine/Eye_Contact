import torch
import torch.nn as nn
from torchsummary import summary


class SubCNN(nn.Module):
    def __init__(self):
        super(SubCNN, self).__init__()
        self.conn1 = nn.Conv2d(1, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(20)
        self.conn2 = nn.Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conn3 = nn.Conv2d(20, 50, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(50)
        self.conn4 = nn.Conv2d(50, 50, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.flatten = nn.Flatten()
        self.act = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(6750, 512)

    def forward(self, val):
        x = self.conn1(val)
        x = self.bn1(x)
        x = self.conn2(x)
        x = self.maxpool(x)
        x = self.conn3(x)
        x = self.bn2(x)
        x = self.conn4(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.act(x)
        x = self.fc1(x)
        return x


def conv_dw(inp, oup, stride, padding=1):
    return nn.Sequential(
        # Depthwise Convolution
        nn.Conv2d(inp, inp, (3, 3), stride, (padding, padding), groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        # Pointwise Convolution
        nn.Conv2d(inp, oup, (1, 1), (1, 1), bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


# Depthwise Separable
class SubCNNDS(nn.Module):
    def __init__(self):
        super(SubCNNDS, self).__init__()
        self.conv_dw1 = conv_dw(1, 20, stride=(1, 1))
        self.conn1 = nn.Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv_dw2 = conv_dw(20, 50, stride=(1, 1))
        self.conn2 = nn.Conv2d(50, 50, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.flatten = nn.Flatten()
        self.act = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(6750, 512)

    def forward(self, val):
        x = self.conv_dw1(val)
        x = self.conn1(x)
        x = self.maxpool(x)
        x = self.conv_dw2(x)
        x = self.conn2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.act(x)
        x = self.fc1(x)
        return x


class Net(nn.Module):
    def __init__(self, subnet):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 2)
        self.subcnn1 = subnet()
        self.subcnn2 = subnet()

    def forward(self, ts1, ts2):
        res1 = self.subcnn1(ts1)
        res2 = self.subcnn2(ts2)
        cated = torch.cat((res1, res2), dim=1)
        x = self.fc1(cated)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    m1 = Net(SubCNN)
    m2 = Net(SubCNNDS)
    summary(m1, [(1, 60, 36), (1, 60, 36)])

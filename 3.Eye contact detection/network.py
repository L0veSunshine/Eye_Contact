import torch
import torch.nn as nn
from torchsummary import summary
from torch import Tensor


class SubCNN(nn.Module):
    def __init__(self):
        super(SubCNN, self).__init__()
        self.conn1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.conn2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conn3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conn4 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conn5 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conn6 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.act = nn.LeakyReLU(inplace=True)
        self.fc1 = nn.Linear(1152, 512)

    def forward(self, val):
        x = self.conn1(val)
        x = self.conn2(x)
        x = self.act(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.conn3(x)
        x = self.conn4(x)
        x = self.act(x)
        x = self.maxpool(x)
        x = self.bn2(x)
        x = self.conn5(x)
        x = self.conn6(x)
        x = self.act(x)
        x = self.maxpool(x)
        x = self.bn3(x)
        x = self.flatten(x)

        x = self.fc1(x)
        return x


class SubCNNA(nn.Module):
    def __init__(self):
        super(SubCNNA, self).__init__()
        self.bn1 = nn.BatchNorm2d(3)
        self.conn1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conn2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conn3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conn4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.conn5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conn6 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
        self.flatten = nn.Flatten()
        self.act = nn.LeakyReLU(inplace=True)
        self.fc1 = nn.Linear(1280, 512)

    def forward(self, val):
        x = self.bn1(val)
        x = self.conn1(x)
        x = self.conn2(x)
        x = self.act(x)
        x = self.maxpool1(x)
        x = self.bn2(x)
        x = self.conn3(x)
        x = self.conn4(x)
        x = self.act(x)
        x = self.maxpool2(x)
        x = self.bn3(x)
        x = self.conn5(x)
        x = self.conn6(x)
        x = self.act(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


def conv_dw(inp, oup, size, stride, padding=1):
    return nn.Sequential(
        # Depthwise Convolution
        nn.Conv2d(inp, inp, (size, size), stride, (padding, padding), groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        # Pointwise Convolution
        nn.Conv2d(inp, oup, (1, 1), stride, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


# Depthwise Separable
class SubCNNDS(nn.Module):
    def __init__(self):
        super(SubCNNDS, self).__init__()
        self.conv_dw1 = conv_dw(3, 20, 5, stride=(1, 1))
        self.conn1 = nn.Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv_dw2 = conv_dw(20, 50, 5, stride=(1, 1))
        self.conn2 = nn.Conv2d(50, 50, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.flatten = nn.Flatten()
        self.act = nn.LeakyReLU(inplace=True)
        self.fc1 = nn.Linear(4200, 512)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(50)

    def forward(self, val):
        x = self.conv_dw1(val)
        x = self.conn1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.conv_dw2(x)
        x = self.conn2(x)
        x = self.maxpool(x)
        x = self.bn2(x)
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

        # self.act = nn.Softmax(dim=1)

    def forward(self, ts1, ts2):
        res1 = self.subcnn1(ts1)
        res2 = self.subcnn2(ts2)
        cated = torch.cat((res1, res2), dim=1)
        x = self.fc1(cated)
        x = self.fc2(x)
        # x = self.act(x)
        return x


class NetA(nn.Module):
    def __init__(self, subnet):
        super(NetA, self).__init__()
        self.fc1 = nn.Linear(1027, 1024)
        self.fc2 = nn.Linear(1024, 2)

        self.subcnn1 = subnet()
        self.subcnn2 = subnet()

        # self.act = nn.Softmax(dim=1)

    def forward(self, ts1, ts2, angle):
        angle = angle.view(angle.size(0), -1)
        res1 = self.subcnn1(ts1)
        res2 = self.subcnn2(ts2)
        cated = torch.cat((res1, res2, angle), dim=1)
        x = self.fc1(cated)
        x = self.fc2(x)
        # x = self.act(x)
        return x


class NetA4(nn.Module):
    def __init__(self, subnet):
        super(NetA4, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1027, 2)
        self.flatten = nn.Flatten()
        self.subcnn1 = subnet()
        self.subcnn2 = subnet()

    def forward(self, ts1, ts2, angle):
        angle = angle.view(angle.size(0), -1)
        res1 = self.subcnn1(ts1)
        res2 = self.subcnn2(ts2)
        cat1 = torch.cat((res1, res2), dim=1)
        x = self.fc1(cat1)
        cat2 = torch.cat((x, angle), dim=1)
        x = self.fc2(cat2)
        return x


if __name__ == '__main__':
    m1 = Net(SubCNN)
    m2 = Net(SubCNNDS)
    m1.cuda()
    m2.cuda()
    summary(m1, [(3, 60, 100), (3, 60, 100)])
    m1a = NetA4(SubCNNA)
    m1a.cuda()
    summary(m1a, [(3, 60, 100), (3, 60, 100), (1, 1, 3)])
    # summary(m1, [(3, 60, 100), (3, 60, 100)])

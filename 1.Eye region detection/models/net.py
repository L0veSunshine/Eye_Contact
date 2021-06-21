import torch
import torch.nn as nn


def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, (3, 3), (stride, stride), (1, 1), bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


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


class SlimNet(nn.Module):
    def __init__(self):
        super(SlimNet, self).__init__()

        self.conv1 = conv_bn(3, 16, 2)
        self.conv2 = conv_dw(16, 32, 1)
        self.conv3 = conv_dw(32, 32, 2)
        self.conv4 = conv_dw(32, 32, 1)
        self.conv5 = conv_dw(32, 64, 2)
        self.conv6 = conv_dw(64, 64, 1)
        self.conv7 = conv_dw(64, 64, 1)
        self.conv8 = conv_dw(64, 64, 1)

        self.conv9 = conv_dw(64, 128, 2)
        self.conv10 = conv_dw(128, 128, 1)
        self.conv11 = conv_dw(128, 128, 1)

        self.conv12 = conv_dw(128, 256, 2)
        self.conv13 = conv_dw(256, 256, 1)

        self.fc = nn.Linear(448, 136)  # 68*2

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        output1 = x8  # [-1, 64, 20, 20]
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        output2 = x11  # [-1, 128, 10, 10]
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        output3 = x13  # [-1, 256, 5, 5]
        output1 = output1.mean(3).mean(2)
        output2 = output2.mean(3).mean(2)
        output3 = output3.mean(3).mean(2)
        output = self.fc(torch.cat((output1, output2, output3), 1))
        return output


if __name__ == '__main__':
    from torchsummary import summary

    model = SlimNet().cuda()
    x = torch.randn(1, 3, 160, 160)
    summary(model, (3, 160, 160))

import torch.nn as nn
from lcnn.models import HT
from lcnn.config import C, M


config_file = "/project/ntimea/l2d2/TRAIN_Github_DOCU_1022/config/wireframe.yaml"
C.update(C.from_yaml(filename=config_file))
M.update(C.model)

print("///////////////// HT_RAL GOOD //////////////////")


# Class used by make_residula (layer1, layer2, layer3, hg, res)
class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # The original input "x" is preserved as "residual" for later use.
        residual = x

        # Three Convolution Operations:
        # Each convolution operation is followed by Batch Normalization and ReLU activation.

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        # Downsampling Check:
        # The "downsample" attribute indicates whether downsampling is required. (layer1 and layer2 YES layer 3 NO)

        if self.downsample is not None:
            residual = self.downsample(x)

        # Residual Addition:
        # The "out" obtained from the convolution operations is added to the "residual."

        out += residual

        return out


# Class used for the 1 by 1 Convolution in the Deconvolution part
class Conv(nn.Module):

    # Initialize a custom convolutional layer.
    # inp_dim: Number of input channels to the convolutional layer.
    # out_dim: Number of output channels produced by the convolution.
    # kernel_size: Size of the convolutional kernel (default: 3x3). BUT IN OUR CASE 1
    # stride: Stride for the convolution operation (default: 1).
    # bn: If True, apply batch normalization (default: False).
    # relu: If True, apply ReLU activation function (default: True). BUT IN OUR CASE FALSE

    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):

        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# The main Class
class HoughTransformNet(nn.Module):

    def __init__(self, block, vote_index, batches_size, depth, num_stacks, num_blocks, num_classes, head):

        super(HoughTransformNet, self).__init__()
        self.num_stacks = num_stacks

        self.inplanes = 64
        self.num_feats = 128
        self.ht_channels = 16
        self.batches = batches_size
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.relu2 = nn.ReLU()
        self.conv = Conv(1, 1, 1, bn=False, relu=False)

        self.h_deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.h_bn1 = nn.BatchNorm2d(8)
        self.h_relu1 = nn.ReLU()
        self.h_deconv2 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.h_bn2 = nn.BatchNorm2d(1)
        self.h_relu2 = nn.ReLU()
        self.h_conv = Conv(1, 1, kernel_size=1, bn=False, relu=False)

        ch = self.num_feats * block.expansion

        self.hg = (self._make_residual(block, self.num_feats, 1))
        self.hg_ht = (HT.CAT_HTIHT(vote_index, inplanes=2 * self.num_feats, outplanes=self.ht_channels))

        self.res = (self._make_residual(block, self.num_feats, num_blocks))
        self.fc = (self._make_fc(ch, ch))
        self.score = (head(ch, num_classes))

    def _make_residual(self, block, planes, blocks, stride=1):

        # Helper method to create a residual block with given parameters.
        # block: The residual block class to be used. (Bottlenec class)
        # planes: Number of output channels for the block.
        # blocks: Number of blocks to be stacked. ALWAYS 1
        # stride: Stride for the convolution operation (default: 1).

        downsample = None

        # expansion = 2 (so if the outputchannel is changed to be twice as big as the inputchannel)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        bn = nn.BatchNorm2d(outplanes)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        out = []

        prea = self.conv1(x)
        prea = self.layer1(prea)
        prea = self.maxpool(prea)
        prea = self.layer2(prea)
        # prea = self.hg(prea)
        
        # # HOUGH TRANSFORM
        prea = self.hg_ht(prea)
        hough_one = prea[1]
        prea = prea[0]

        prea = self.res(prea)
        # #y = self.fc(y)
        # score = self.score(prea)
        # out.append(score)
        # prea = prea

        prea = self.layer3(prea)

        # Hough Transform
        #prea = self.hg(prea)

        prea = self.hg_ht(prea)
        hough_one = prea[1]
        prea = prea[0]
        prea = self.res(prea)
        #y = self.fc(y)
        # score = self.score(prea)
        # out.append(score)

        # Deconv Hough
        hough_line_dec1 = self.h_deconv1(hough_one)
        hough_line_dec1 = self.h_bn1(hough_line_dec1)
        hough_line_dec1 = self.h_relu1(hough_line_dec1)

        hough_line_dec2 = self.h_deconv2(hough_line_dec1)
        hough_line_dec2 = self.h_bn2(hough_line_dec2)
        hough_line_dec2 = self.h_relu2(hough_line_dec2)

        hough_line_detected = self.h_conv(hough_line_dec2)

        # mask = hough_line_detected < 0
        # hough_line_detected[mask] = 0
        prea = prea

        # Deconv
        line_dec1 = self.deconv1(prea)
        line_dec1 = self.bn1(line_dec1)
        line_dec1 = self.relu1(line_dec1)

        line_dec2 = self.deconv2(line_dec1)
        line_dec2 = self.bn2(line_dec2)
        line_dec2 = self.relu2(line_dec2)

        line_detected = self.conv(line_dec2)

        # line_detected = 512*512
        # hough_line_detected 728*240

        return line_detected, hough_line_detected


def ht(**kwargs):
    # batches_size: 1 , depth: 4, num_stacks: 4, num_blocks: 1, num_classes: 5

    model = HoughTransformNet(
        Bottleneck2D,

        batches_size=kwargs["batches_size"],
        depth=kwargs["depth"],

        head=kwargs.get("head", lambda c_in, c_out: nn.Conv2D(c_in, c_out, 1)),
        num_stacks=kwargs["num_stacks"],
        num_blocks=kwargs["num_blocks"],
        num_classes=kwargs["num_classes"],
        vote_index=kwargs["vote_index"],
    )
    return model
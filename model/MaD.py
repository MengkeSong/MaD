import torch
import torch.nn as nn
import torchvision.models as models
from ResNet import ResNet50
from torch.nn import functional as F
from torchvision.models import resnet50


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# Global Contextual module
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation(nn.Module):

    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)


    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(x1) * x2
        x3_1 = self.conv_upsample2(x1) * self.conv_upsample3(x2) * x3
        x2_2 = torch.cat((x2_1, self.conv_upsample4(x1_1)), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)

        return x3_2

# aggregation of the high-level features
class aggregation_init(nn.Module):

    def __init__(self, channel):
        super(aggregation_init, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        return x3_2


# aggregation of the low-level(student) features
class aggregation_final(nn.Module):

    def __init__(self, channel):
        super(aggregation_final, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x1)) \
               * self.conv_upsample3(x2) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)

        return x3_2

class GloRe(nn.Module):
    def __init__(self, in_channels):
        super(GloRe, self).__init__()
        self.N = in_channels // 4
        self.S = in_channels // 2
        
        self.theta = nn.Conv2d(in_channels, self.N, 1, 1, 0, bias=False)
        self.phi = nn.Conv2d(in_channels, self.S, 1, 1, 0, bias=False)
        
        self.relu = nn.ReLU()
        
        self.node_conv = nn.Conv1d(self.N, self.N, 1, 1, 0, bias=False)
        self.channel_conv = nn.Conv1d(self.S, self.S, 1, 1, 0, bias=False)
        
        self.conv_2 = nn.Conv2d(self.S, in_channels, 1, 1, 0, bias=False)
        
    def forward(self, x):
        batch, C, H, W = x.size()
        L = H * W
        
        B = self.theta(x).view(-1, self.N, L)

        phi = self.phi(x).view(-1, self.S, L)
        phi = torch.transpose(phi, 1, 2)

        V = torch.bmm(B, phi)
        V = self.relu(self.node_conv(V))
        V = self.relu(self.channel_conv(torch.transpose(V, 1, 2)))
        
        y = torch.bmm(torch.transpose(B, 1, 2), torch.transpose(V, 1, 2))
        y = y.view(-1, self.S, H, W)
        y = self.conv_2(y)
        
        return x + y

# MaDNet
class MaDNet(nn.Module):
    def __init__(self, channel=32):
        super(MaDNet, self).__init__()

        # Backbone model
        self.resnet = ResNet50('rgb')
        self.resnet_depth = ResNet50('rgbd')

        self.convd1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=True)
        self.convd2 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=True)
        self.convd3 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=True)
        self.gcn1 = GloRe(512*3)

        self.convd4 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=True)
        self.convd5 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=True)
        # self.convd6 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=True)
        self.gcn2 = GloRe(512*3)



        self.rfb2_1 = GCM(512, channel)
        self.rfb3_1 = GCM(512, channel)
        self.rfb4_1 = GCM(512, channel)
        self.agg0 = aggregation(channel)

        self.rfb2_11 = GCM(512, channel)
        self.rfb3_11 = GCM(512, channel)
        self.rfb4_11 = GCM(512, channel)
        self.agg1 = aggregation_init(channel)

        # Decoder
        self.rfb0_2 = GCM(64, channel)
        self.rfb1_2 = GCM(256, channel)
        self.rfb5_2 = GCM(512, channel)
        self.agg2 = aggregation_final(channel)

        # upsample function
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Components of DEM module
        self.atten_depth_channel_0 = ChannelAttention(64)
        self.atten_depth_channel_1 = ChannelAttention(256)
        self.atten_depth_channel_2 = ChannelAttention(512)
        self.atten_depth_channel_3_1 = ChannelAttention(1024)
        self.atten_depth_channel_4_1 = ChannelAttention(2048)

        self.atten_depth_spatial_0 = SpatialAttention()
        self.atten_depth_spatial_1 = SpatialAttention()
        self.atten_depth_spatial_2 = SpatialAttention()
        self.atten_depth_spatial_3_1 = SpatialAttention()
        self.atten_depth_spatial_4_1 = SpatialAttention()

        # Components of PTM module
        self.inplanes = 32 * 2
        self.deconv1 = self._make_transpose(TransBasicBlock, 32 * 2, 3, stride=2)
        self.inplanes = 32
        self.deconv2 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.agant1 = self._make_agant_layer(32 * 3, 32 * 2)
        self.agant2 = self._make_agant_layer(32 * 2, 32)
        self.out0_conv = nn.Conv2d(32 * 3, 1, kernel_size=1, stride=1, bias=True)
        self.out1_conv = nn.Conv2d(32 * 2, 1, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(32 * 1, 1, kernel_size=1, stride=1, bias=True)

        self.conv0 = BasicConv2d(4*channel, 2*channel, 3, padding=1)
        self.conv1 = BasicConv2d(2*channel, 8 * channel, 3, padding=1,stride=2)
        self.conv2 = BasicConv2d(8 * channel, 16 * channel, 3, padding=1,stride=2)
        self.conv3 = BasicConv2d(16 * channel, 32 * channel, 3, padding=1,stride=2)
        self.conv4 = BasicConv2d(32 * channel, 64 * channel, 3, padding=1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.pooling_1 = nn.AdaptiveAvgPool2d(1)
        self.pooling_22 = nn.AdaptiveAvgPool2d(22)
        self.pooling_11 = nn.AdaptiveAvgPool2d(11)

        self.fc = BasicConv2d(3 * channel, 64 * channel, 1, padding=0)
        self.fc1 = BasicConv2d(3 * channel, 32 * channel,1, padding=0)
        self.fc2 = BasicConv2d(3 * channel, 16 * channel,1, padding=0)

        self.fc3 = BasicConv2d(3 * channel, 16 * channel, 1, padding=0)
        self.fc4 = BasicConv2d(3 * channel, 8 * channel, 1, padding=0)
        self.fc5 = BasicConv2d(3 * channel, 2 * channel, 1, padding=0)

        self.convdownsample = BasicConv2d(3*channel, 1, 1, padding=0)


        if self.training:
            self.initialize_weights()

    def forward(self, x, x_depth):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_depth = self.resnet_depth.conv1(x_depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x_depth = self.resnet_depth.maxpool(x_depth)

        # layer0 merge
        x_cat = self.conv0(torch.cat((x, x_depth), 1))
        temp_c = x_cat.mul(self.atten_depth_channel_0(x_cat))
        x_cat = temp_c.mul(self.atten_depth_spatial_0(temp_c))

        temp = x_depth.mul(self.atten_depth_channel_0(x_depth))
        temp = temp.mul(self.atten_depth_spatial_0(temp))
        x = x + temp
        # layer0 merge end

        x1 = self.resnet.layer1(x)
        x1_depth = self.resnet_depth.layer1(x_depth)

        # layer1 merge
        x_cat1 = self.conv1(x_cat)
        temp = x_cat1.mul(self.atten_depth_channel_1(x_cat1))
        x_cat1 = temp.mul(self.atten_depth_spatial_1(temp))

        temp = x1_depth.mul(self.atten_depth_channel_1(x1_depth))
        temp = temp.mul(self.atten_depth_spatial_1(temp))
        x1 = x1 + temp
        # layer1 merge end

        x2 = self.resnet.layer2(x1)
        x2_depth = self.resnet_depth.layer2(x1_depth)

        # layer2 merge
        x_cat2 = self.conv2(x_cat1)
        temp = x_cat2.mul(self.atten_depth_channel_2(x_cat2))
        x_cat2 = temp.mul(self.atten_depth_spatial_2(temp))

        temp = x2_depth.mul(self.atten_depth_channel_2(x2_depth))
        temp = temp.mul(self.atten_depth_spatial_2(temp))
        x2 = x2 + temp
        # layer2 merge end

        x2_1 = x2

        x3_1 = self.resnet.layer3_1(x2_1)
        x3_1_depth = self.resnet_depth.layer3_1(x2_depth)

        # layer3_1 merge
        x_cat3 = self.conv3(x_cat2)
        temp = x_cat3.mul(self.atten_depth_channel_3_1(x_cat3))
        x_cat3 = temp.mul(self.atten_depth_spatial_3_1(temp))

        temp = x3_1_depth.mul(self.atten_depth_channel_3_1(x3_1_depth))
        temp = temp.mul(self.atten_depth_spatial_3_1(temp))
        x3_1 = x3_1 + temp
        # layer3_1 merge end

        x4_1 = self.resnet.layer4_1(x3_1)
        x4_1_depth = self.resnet_depth.layer4_1(x3_1_depth)

        # layer4_1 merge
        x_cat4 = self.conv4(x_cat3)
        temp = x_cat4.mul(self.atten_depth_channel_4_1(x_cat4))
        x_cat4 = temp.mul(self.atten_depth_spatial_4_1(temp))

        temp = x4_1_depth.mul(self.atten_depth_channel_4_1(x4_1_depth))
        temp = temp.mul(self.atten_depth_spatial_4_1(temp))
        x4_1 = x4_1 + temp
        # layer4_1 merge end

        #Graph weighting
        #GCN1
        x4_1_depth_g = self.convd1(x4_1_depth)
        x_cat4_g = self.convd2(x_cat4)
        x4_1_g = self.convd3(x4_1)
        gcn_out1 = self.gcn1(torch.cat((x4_1_depth_g,x_cat4_g,x4_1_g),1))
        x4_1_depth_gg = gcn_out1[:, :512, :, :]
        x_cat4_gg = gcn_out1[:, 512:1024, :, :]
        x4_1_gg = gcn_out1[:, 1024:1536, :, :]

        #FE1
        x4_1_depth_ggg = self.rfb2_1(x4_1_depth_gg)
        x_cat4_ggg     = self.rfb3_1(x_cat4_gg)
        x4_1_ggg       = self.rfb4_1(x4_1_gg)

        out = self.agg0(x4_1_depth_ggg, x_cat4_ggg, x4_1_ggg)

        out_1 = self.pooling(out)
        weighting = torch.softmax(out_1,1)

        weighting_2_1 = self.fc(weighting)
        x4_1_w = x4_1.mul(weighting_2_1)
        weight_3_1 = self.fc1(weighting)
        x3_1_w = x3_1.mul(weight_3_1)
        weight_4_1 = self.fc2(weighting)
        x2_1_w = x2_1.mul(weight_4_1)

        #GCN2
        x4_1_w_g = self.upsample4(self.convd4(x4_1_w))
        x3_1_w_g = self.upsample2(self.convd5(x3_1_w))
        x2_1_w_g = x2_1_w
        gcn_out2 = self.gcn2(torch.cat((x4_1_w_g,x3_1_w_g,x2_1_w_g),1))
        x4_1_w_gg = gcn_out2[:, :512, :, :]
        x3_1_w_gg = gcn_out2[:, 512:1024, :, :]
        x2_1_w_gg = gcn_out2[:, 1024:1536, :, :]


        # FE2
        x2_1_w_ggg = self.rfb2_11(x2_1_w_gg)
        x3_1_w_ggg = self.rfb3_11(x3_1_w_gg)
        x4_1_w_ggg = self.rfb4_11(x4_1_w_gg)


        x4_1_w_ggg = self.pooling_11(x4_1_w_ggg)
        x3_1_w_ggg = self.pooling_22(x3_1_w_ggg)

        out_2 = self.agg1(x4_1_w_ggg, x3_1_w_ggg, x2_1_w_ggg)

        out_3 = self.pooling_1(out_2)
        weighter_1 = torch.softmax(out_3,1)

        weighting_2 = self.fc3(weighter_1)
        x4_1_ww = x2.mul(weighting_2)
        weighting_1 = self.fc4(weighter_1)
        x3_1_ww = x1.mul(weighting_1)
        weighting_0 = self.fc5(weighter_1)
        x2_1_ww = x.mul(weighting_0)

        #Decoder
        x5_2 = self.rfb5_2(x4_1_ww)
        x1_2 = self.rfb1_2(x3_1_ww)
        x0_2 = self.rfb0_2(x2_1_ww)
        y = self.agg2(x5_2, x1_2, x0_2)

        # PTM module
        y = self.agant1(y)
        y = self.deconv1(y)
        y = self.agant2(y)
        y = self.deconv2(y)
        y = self.out2_conv(y)

        return y

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    # initialize the weights
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_depth.state_dict().items():
            if k == 'conv1.weight':
                all_params[k] = torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_depth.state_dict().keys())
        self.resnet_depth.load_state_dict(all_params)


import torch
import torch.nn as nn
import torch.nn.functional as F


class DFEB(nn.Module):
    def __init__(self, in_channels):
        super(DFEB, self).__init__()


        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)


        self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.k_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)


        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)


        self.conv3 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.gap_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)


        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x_conv = F.relu(self.conv1(x))  # (batch_size, in_channels, H, W)


        Q = self.q_conv(x_conv)
        K = self.k_conv(x_conv)
        V = self.v_conv(x_conv)


        attn_weights = F.softmax(torch.matmul(Q.view(Q.size(0), Q.size(1), -1),
                                              K.view(K.size(0), K.size(1), -1).transpose(1, 2)), dim=-1)


        attn_output = torch.matmul(attn_weights, V.view(V.size(0), V.size(1), -1))
        attn_output = attn_output.view(x.size(0), x.size(1), *x.size()[2:])
        attn_output = F.relu(self.conv2(attn_output))


        concat_output = torch.cat((x_conv, attn_output), dim=1)  # (batch_size, in_channels * 2, H, W)
        x_after_concat = self.conv3(concat_output)  # (batch_size, in_channels, H, W)


        gap_output = F.adaptive_avg_pool2d(x_after_concat, (1, 1))  # (batch_size, in_channels, 1, 1)
        gap_output = self.gap_conv(gap_output)
        gap_output = self.sigmoid(gap_output)


        x_scaled = x_after_concat * gap_output.expand_as(x_after_concat)


        output = x + x_scaled  # (batch_size, in_channels, H, W)

        return output


class DownsampleConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleConvModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_resample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)

    def forward(self, x):

        residual = self.conv_resample(x)


        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x += residual
        return x


class UpsampleConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleConvModule, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.upsample(x)

        return x


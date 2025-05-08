import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from model.partModel import *
from model.capsule import *
from utils import *
import time
class CapUBS(nn.Module):
    def __init__(self,
                 image_width,
                 image_height,
                 image_channels,
                 num_output_units,
                 output_unit_size,
                 num_iterations):
        super(CapUBS, self).__init__()

        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels

        # X
        self.DFEB_1 = DFEB(self.image_channels)
        self.DownConv_1 = DFEB(self.image_channels)
        self.DFEB_2 = DFEB(self.image_channels)
        self.DownConv_2 = DFEB(self.image_channels)

        # X_var
        self.DFEB_3 = DFEB(self.image_channels)
        self.DownConv_3 = DFEB(self.image_channels)
        self.DFEB_4 = DFEB(self.image_channels)


        # Capsule
        self.capsule_net = CapsuleNetwork(
                       image_width=self.image_width,
                       image_height=self.image_height,
                       image_channels=self.image_channels,
                       conv_inputs=self.image_channels,
                       conv_outputs=self.image_channels,
                       num_primary_units=self.image_channels,
                       primary_unit_size=self.image_channels,
                       num_output_units=num_output_units,  # one for each MNIST digit
                       output_unit_size=output_unit_size,
                       num_iterations=num_iterations
                       )
        # recovery
        self.DFEB_5 = DFEB(self.image_channels)
        self.conv_5 = nn.Conv2d(self.image_channels, self.image_channels * 2, kernel_size=1, stride=1, padding=0)

        self.DFEB_6 = DFEB(self.image_channels * 2)
        self.conv_6 = nn.Conv2d(self.image_channels * 2, self.image_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, x_var):

        # time1 = time.time()
        # Encoder
        x_var_fea = self.DFEB_3(x_var)
        x_fea_skip_1 = self.DFEB_1(x) + x_var_fea
        x_fea = self.DownConv_1(x_fea_skip_1)

        x_var_fea = self.DownConv_3(x_var_fea)
        x_var_fea = self.DFEB_4(x_var_fea)
        x_fea_skip_2 = self.DFEB_2(x_fea) + x_var_fea
        # time2 = time.time()
        # print(f'time of feature extra is {time2 - time1}')
        # Capsule
        band_attention, cap_fea = self.capsule_net(x_fea_skip_2)
        # time3 = time.time()
        # print(f'time of capsule net is {time3 - time2}')

        # Decoder
        de_fea = self.DFEB_5(cap_fea)
        de_fea = self.conv_5(de_fea)

        de_fea = self.DFEB_6(de_fea)
        x_rec = self.conv_6(de_fea)
        # time4 = time.time()
        # print(f'time of reconstruct is {time4 - time3}')

        return band_attention, x_rec

    def loss(self, band_attention, x, x_rec):

        x_band = band_attention * x

        Loss_MSE = F.mse_loss(x, x_rec)
        Loss_SA = SALoss(x, x_rec)
        Loss_rec = Loss_MSE + 1e-2 * Loss_SA


        Loss_var = 1 / torch.var(x_band)
        Loss_entropy = 1 / entropyLoss(x_band)
        Loss_sparsity = sparsityLoss(band_attention)
        Loss_bs = 1e-4 * Loss_var + Loss_entropy + 1e-2 * Loss_sparsity

        total_loss = 1e-1 * Loss_rec + 1e-3 * Loss_bs

        return total_loss, 1e-1 * Loss_rec, 1e-3 * Loss_bs


if __name__ == '__main__':

    batch_size = 8
    img_channels = 10
    img_height = 28
    img_width = 28
    num_output_units = 10
    output_unit_size = 16
    num_iterations = 10



    model = CapUBS(image_width=img_width,
                             image_height=img_height,
                             image_channels=img_channels,
                             num_output_units=num_output_units,  # one for each MNIST digit
                             output_unit_size=output_unit_size,
                             num_iterations =num_iterations
                            )

    optimizer = Adam(model.parameters(), lr=0.001)


    input_data = torch.randn(batch_size, img_channels, img_width, img_height)  # (batch_size, channels, height, width)


    band_attention, x_rec = model(input_data, input_data)
    print(SALoss(input_data, input_data))
    print(band_attention.shape)
    print(x_rec.shape)
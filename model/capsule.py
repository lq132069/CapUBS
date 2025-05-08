import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from .partModel import  *

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x):
        return F.relu(self.conv(x))

class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()

        self.conv0 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.conv0(x)

class PrimaryCapsuleLayer(nn.Module):
    def __init__(self, in_units, in_size, out_units, out_size):
        super(PrimaryCapsuleLayer, self).__init__()
        self.in_units = in_units
        self.in_channels = in_size
        self.out_units = out_units
        self.out_size = out_size

        def create_conv_unit(unit_idx):
            unit = ConvUnit(in_channels=in_size)
            self.add_module("unit_" + str(unit_idx), unit)
            return unit

        self.units = [create_conv_unit(i) for i in range(self.out_units)]

    @staticmethod
    def squash(s):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        # Get output for each unit.
        # Each will be (batch, channels, height, width).
        u = [self.units[i](x) for i in range(self.out_units)]  # 8个torch.Size([8, 32, 6, 6])

        # Stack all unit outputs (batch, unit, channels, height, width).
        u = torch.stack(u, dim=1) # torch.Size([8, 8, 32, 6, 6])
        # Flatten to (batch, unit, output).
        u = u.view(x.size(0), self.out_units, -1)

        # Return squashed outputs.
        return self.squash(u)

class CapsuleLayer(nn.Module):
    def __init__(self, in_units, in_size, out_units, out_size, num_iterations):
        super(CapsuleLayer, self).__init__()
        self.in_units = in_units
        self.in_channels = in_size
        self.out_units = out_units
        self.out_size = out_size
        self.num_iterations = num_iterations
        self.W = nn.Parameter(torch.randn(1, in_size, out_units, out_size, in_units))


    def forward(self, x):
        batch_size = x.size(0)
        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2)
        # (batch, features, in_units) -> (batch, features, num_units, in_units, 1)
        x = torch.stack([x] * self.out_units, dim=2).unsqueeze(4)

        # (batch, features, in_units, unit_size, num_units)
        W = torch.cat([self.W] * batch_size, dim=0)
        # Transform inputs by weight matrix.
        # (batch_size, features, num_units, unit_size, 1)
        u_hat = torch.matmul(W, x)
        # Initialize routing logits to zero.
        b_ij = torch.ones(1, self.in_channels, self.out_units, 1).cpu()
        # Iterative routing.
        for iteration in range(self.num_iterations):
            # Convert routing logits to softmax.
            # (batch, features, num_units, 1, 1)
            c_ij = F.softmax(b_ij, dim=1)
            c_ij_r = c_ij.detach()
            # c_ij = b_ij
            # c_ij_r = c_ij
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, num_units, unit_size, 1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # (batch_size, 1, num_units, unit_size, 1)
            v_j = self.squash(s_j)

            # (batch_size, features, num_units, unit_size, 1)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)

            # (1, features, num_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update b_ij (routing)
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1)

    def squash(self, s):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

class ReconstructionLayer(nn.Module):
    def __init__(self, image_width, image_height, image_channels, num_output_units, output_unit_size):
        super(ReconstructionLayer, self).__init__()

        self.image_channels = image_channels
        self.image_width = image_width
        self.image_height = image_height

        reconstruction_size = image_width * image_height * image_channels

        self.fc = nn.Sequential(
                    nn.Linear(num_output_units*output_unit_size, int((reconstruction_size * 2) / 3)),
                    nn.Linear(int((reconstruction_size * 2) / 3), int((reconstruction_size * 3) / 2)),
                    nn.Linear(int((reconstruction_size * 3) / 2), reconstruction_size)
                    )

        self.conv = nn.ConvTranspose2d(image_channels, image_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, self.image_channels, self.image_height, self.image_width)
        return x

class CapsuleNetwork(nn.Module):
    def __init__(self,
                 image_width,
                 image_height,
                 image_channels,
                 conv_inputs,
                 conv_outputs,
                 num_primary_units,
                 primary_unit_size,
                 num_output_units,
                 output_unit_size,
                 num_iterations):
        super(CapsuleNetwork, self).__init__()

        self.image_channels = image_channels
        self.image_width = image_width
        self.image_height = image_height

        self.conv_layer = ConvLayer(
            in_channels=conv_inputs,
            out_channels=conv_outputs
        )

        self.primary_capsule_layer = PrimaryCapsuleLayer(
            in_units=0,
            in_size=conv_outputs,
            out_units=num_primary_units,
            out_size=primary_unit_size
        )

        self.capsule_layer = CapsuleLayer(
            in_units=num_primary_units,
            in_size=primary_unit_size,
            out_units=num_output_units,
            out_size=output_unit_size,
            num_iterations=num_iterations
        )

        self.reconstruction_layer = ReconstructionLayer(
                 image_width=image_width,
                 image_height=image_height,
                 image_channels=image_channels,
                 num_output_units=num_output_units,
                 output_unit_size=output_unit_size)

        self.fixed_c_ij = None

    def forward(self, x):

        x = self.conv_layer(x)
        x = self.primary_capsule_layer(x)
        v_j = self.capsule_layer(x)

        band_attention = torch.norm(v_j, dim=-2).unsqueeze(2)  # (batch_size, img_channels, 1)

        reconstructed = self.reconstruction_layer(v_j * band_attention)  # v_j:torch.Size([8, 10, 16, 1])

        return band_attention, reconstructed



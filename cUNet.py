import torch
import torch.nn as nn

from cUNet_bb import *
from utils import get_nonlinearity


# Second-order ODE UNet
class ConvSODEUNet(nn.Module):
    def __init__(
        self,
        num_filters,
        output_dim=1,
        time_dependent=False,
        non_linearity="softplus",
        tol=1e-3,
        adjoint=False,
        method="rk4"
    ):
        """
        ConvSODEUNet (Second order ODE UNet)
        Args:
            num_filters (int): number of filters for first conv layer
            output_dim (int): how many feature maps the network outputs
            time_dependent (bool): whether to concat the time as a feature map before the convs
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
            tol (float): tolerance to be used for ODE solver
            adjoint (bool): whether to use the adjoint method to calculate the gradients
        """
        super(ConvSODEUNet, self).__init__()
        nf = num_filters
        self.method = method
        print(f"Solver: {method}")

        self.input_1x1 = nn.Conv2d(3, nf, 1, 1)
        self.initial_velocity = InitialVelocity(nf, non_linearity)

        ode_down1 = ConvSODEFunc(nf * 2, time_dependent, non_linearity)
        self.odeblock_down1 = ODEBlock(ode_down1, tol=tol, adjoint=adjoint)
        self.conv_down1_2 = nn.Conv2d(nf * 2, nf * 4, 1, 1)

        ode_down2 = ConvSODEFunc(nf * 4, time_dependent, non_linearity)
        self.odeblock_down2 = ODEBlock(ode_down2, tol=tol, adjoint=adjoint)
        self.conv_down2_3 = nn.Conv2d(nf * 4, nf * 8, 1, 1)

        ode_down3 = ConvSODEFunc(nf * 8, time_dependent, non_linearity)
        self.odeblock_down3 = ODEBlock(ode_down3, tol=tol, adjoint=adjoint)
        self.conv_down3_4 = nn.Conv2d(nf * 8, nf * 16, 1, 1)

        ode_down4 = ConvSODEFunc(nf * 16, time_dependent, non_linearity)
        self.odeblock_down4 = ODEBlock(ode_down4, tol=tol, adjoint=adjoint)
        self.conv_down4_embed = nn.Conv2d(nf * 16, nf * 32, 1, 1)

        ode_embed = ConvSODEFunc(nf * 32, time_dependent, non_linearity)
        self.odeblock_embedding = ODEBlock(ode_embed, tol=tol, adjoint=adjoint)
        self.conv_up_embed_1 = nn.Conv2d(nf * 32 + nf * 16, nf * 16, 1, 1)

        ode_up1 = ConvSODEFunc(nf * 16, time_dependent, non_linearity)
        self.odeblock_up1 = ODEBlock(ode_up1, tol=tol, adjoint=adjoint)
        self.conv_up1_2 = nn.Conv2d(nf * 16 + nf * 8, nf * 8, 1, 1)

        ode_up2 = ConvSODEFunc(nf * 8, time_dependent, non_linearity)
        self.odeblock_up2 = ODEBlock(ode_up2, tol=tol, adjoint=adjoint)
        self.conv_up2_3 = nn.Conv2d(nf * 8 + nf * 4, nf * 4, 1, 1)

        ode_up3 = ConvSODEFunc(nf * 4, time_dependent, non_linearity)
        self.odeblock_up3 = ODEBlock(ode_up3, tol=tol, adjoint=adjoint)
        self.conv_up3_4 = nn.Conv2d(nf * 4 + nf * 2, nf * 2, 1, 1)

        ode_up4 = ConvSODEFunc(nf * 2, time_dependent, non_linearity)
        self.odeblock_up4 = ODEBlock(ode_up4, tol=tol, adjoint=adjoint)

        self.classifier = nn.Conv2d(nf * 2, output_dim, 1)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x):
        x = self.initial_velocity(x)

        features1 = self.odeblock_down1(x,  method=self.method)  # 512
        x = self.non_linearity(self.conv_down1_2(features1))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        features2 = self.odeblock_down2(x,  method=self.method)  # 256
        x = self.non_linearity(self.conv_down2_3(features2))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        features3 = self.odeblock_down3(x,  method=self.method)  # 128
        x = self.non_linearity(self.conv_down3_4(features3))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        features4 = self.odeblock_down4(x,  method=self.method)  # 64
        x = self.non_linearity(self.conv_down4_embed(features4))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        x = self.odeblock_embedding(x,  method=self.method)  # 32

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features4), dim=1)
        x = self.non_linearity(self.conv_up_embed_1(x))
        x = self.odeblock_up1(x,  method=self.method)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features3), dim=1)
        x = self.non_linearity(self.conv_up1_2(x))
        x = self.odeblock_up2(x,  method=self.method)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features2), dim=1)
        x = self.non_linearity(self.conv_up2_3(x))
        x = self.odeblock_up3(x,  method=self.method)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features1), dim=1)
        x = self.non_linearity(self.conv_up3_4(x))
        x = self.odeblock_up4(x,  method=self.method)

        pred = self.classifier(x)
        return pred
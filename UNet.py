import torch
import torch.nn as nn

from utils import sinusoidal_embedding


class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out


class MyUNet(nn.Module):
    """
    Custom U-Net with 4 down-sample parts, a bottleneck in the middle of the network, and 3 up-sample steps with
    the usual U-Net residual connections (concatenations).

    Args:
        n_steps (int): The number of diffusion steps.
        time_emb_dim (int): The dimensionality of the time embedding.

    Attributes:
        time_embed (nn.Embedding): The sinusoidal positional embedding.
        te1 (nn.Sequential): The positional embedding MLP for the first half.
        b1 (nn.Sequential): The first half block of the U-Net.
        down1 (nn.Conv2d): The first downsample operation.
        te2 (nn.Sequential): The positional embedding MLP for the second half.
        b2 (nn.Sequential): The second half block of the U-Net.
        down2 (nn.Conv2d): The second downsample operation.
        te3 (nn.Sequential): The positional embedding MLP for the third half.
        b3 (nn.Sequential): The third half block of the U-Net.
        down3 (nn.Sequential): The third downsample operation.
        te_mid (nn.Sequential): The positional embedding MLP for the bottleneck.
        b_mid (nn.Sequential): The bottleneck block of the U-Net.
        up1 (nn.Sequential): The first upsample operation.
        te4 (nn.Sequential): The positional embedding MLP for the fourth half.
        b4 (nn.Sequential): The fourth half block of the U-Net.
        up2 (nn.ConvTranspose2d): The second upsample operation.
        te5 (nn.Sequential): The positional embedding MLP for the fifth half.
        b5 (nn.Sequential): The fifth half block of the U-Net.
        up3 (nn.ConvTranspose2d): The third upsample operation.
        te_out (nn.Sequential): The positional embedding MLP for the output.
        b_out (nn.Sequential): The output block of the U-Net.
        conv_out (nn.Conv2d): The final convolutional layer.

    """

    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MyUNet, self).__init__()

        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 28, 28), 1, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 14, 14), 10, 20),
            MyBlock((20, 14, 14), 20, 20),
            MyBlock((20, 14, 14), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 7, 7), 20, 40),
            MyBlock((40, 7, 7), 40, 40),
            MyBlock((40, 7, 7), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, 3, 3), 40, 20),
            MyBlock((20, 3, 3), 20, 20),
            MyBlock((20, 3, 3), 20, 40)
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, 7, 7), 80, 40),
            MyBlock((40, 7, 7), 40, 20),
            MyBlock((20, 7, 7), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 14, 14), 40, 20),
            MyBlock((20, 14, 14), 20, 10),
            MyBlock((10, 14, 14), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 28, 28), 20, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t):
        """
        Forward pass of the MyUNet model.

        Args:
            x (torch.Tensor): The input tensor.
            t (torch.Tensor): The time tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))

        out5 = torch.cat((out2, self.up2(out4)), dim=1)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))

        out = torch.cat((out1, self.up3(out5)), dim=1)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        """
        Utility function that creates a one-layer MLP which will be used to map a positional embedding.

        Args:
            dim_in (int): The input dimensionality.
            dim_out (int): The output dimensionality.

        Returns:
            nn.Sequential: The MLP model.

        """
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

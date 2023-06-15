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

  def __init__(self, n_steps=1000, time_emb_dim=100):
    super(MyUNet, self).__init__()

    self.time_embed = nn.Embedding(n_steps, time_emb_dim)
    self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
    self.time_embed.requires_grad_(False)

    self.te1 = self._make_te(time_emb_dim, 3)
    self.b1 = nn.Sequential(
        MyBlock((3, 96, 96), 3, 64),
        MyBlock((64, 96, 96), 64, 64)
    )
    self.down1 = nn.Conv2d(64, 64, 4, 2, 1)

    self.te2 = self._make_te(time_emb_dim, 64)
    self.b2 = nn.Sequential(
        MyBlock((64, 48, 48), 64, 128),
        MyBlock((128, 48, 48), 128, 128)
    )
    self.down2 = nn.Conv2d(128, 128, 4, 2, 1)

    self.te3 = self._make_te(time_emb_dim, 128)
    self.b3 = nn.Sequential(
        MyBlock((128, 24, 24), 128, 256),
        MyBlock((256, 24, 24), 256, 256)
    )
    self.down3 = nn.Conv2d(256, 256, 4, 2, 1)

    self.te4 = self._make_te(time_emb_dim, 256)
    self.b4 = nn.Sequential(
        MyBlock((256, 12, 12), 256, 512),
        MyBlock((512, 12, 12), 512, 512)
    )
    self.down4 = nn.Conv2d(512, 512, 4, 2, 1)

    # Bottleneck
    self.te_mid = self._make_te(time_emb_dim, 512)
    self.b_mid = nn.Sequential(
        MyBlock((512, 6, 6), 512, 1024),
        MyBlock((1024, 6, 6), 1024, 1024)
    )

    # Decoding Path
    self.up1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
    self.te5 = self._make_te(time_emb_dim, 1024)
    self.b5 = nn.Sequential(
        MyBlock((1024, 12, 12), 1024, 512),
        MyBlock((512, 12, 12), 512, 512)
    )

    self.up2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
    self.te6 = self._make_te(time_emb_dim, 512)
    self.b6 = nn.Sequential(
        MyBlock((512, 24, 24), 512, 256),
        MyBlock((256, 24, 24), 256, 256)
    )

    self.up3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
    self.te7 = self._make_te(time_emb_dim, 256)
    self.b7 = nn.Sequential(
        MyBlock((256, 48, 48), 256, 128),
        MyBlock((128, 48, 48), 128, 128)
    )

    self.up4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
    self.te_out = self._make_te(time_emb_dim, 128)
    self.b_out = nn.Sequential(
        MyBlock((128, 96, 96), 128, 64),
        MyBlock((64, 96, 96), 64, 64)
    )

    self.conv_out = nn.Conv2d(64, 3, 3, 1, 1)

  def forward(self, x, t):
    t = self.time_embed(t)
    n = len(x)

    te1 = self.te1(t).reshape(n, -1, 1, 1)
    out1 = self.b1(x + te1)
    d1 = self.down1(out1)

    te2 = self.te2(t).reshape(n, -1, 1, 1)
    out2 = self.b2(d1 + te2)
    d2 = self.down2(out2)

    te3 = self.te3(t).reshape(n, -1, 1, 1)
    out3 = self.b3(d2 + te3)
    d3 = self.down3(out3)

    te4 = self.te4(t).reshape(n, -1, 1, 1)
    out4 = self.b4(d3 + te4)
    d4 = self.down4(out4)

    te_mid = self.te_mid(t).reshape(n, -1, 1, 1)
    out_mid = self.b_mid(d4 + te_mid)

    up1 = self.up1(out_mid)
    te5 = self.te5(t).reshape(n, -1, 1, 1)
    out5 = self.b5(torch.cat((out4, up1), dim=1) + te5)

    up2 = self.up2(out5)
    te6 = self.te6(t).reshape(n, -1, 1, 1)
    out6 = self.b6(torch.cat((out3, up2), dim=1) + te6)

    up3 = self.up3(out6)
    te7 = self.te7(t).reshape(n, -1, 1, 1)
    out7 = self.b7(torch.cat((out2, up3), dim=1) + te7)

    up4 = self.up4(out7)
    te_out = self.te_out(t).reshape(n, -1, 1, 1)
    out = self.b_out(torch.cat((out1, up4), dim=1) + te_out)

    out = self.conv_out(out)

    return out


  def _make_te(self, dim_in, dim_out):
    return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.SiLU(),
        nn.Linear(dim_out, dim_out)
    )

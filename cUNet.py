class MyContinuousUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MyContinuousUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            ODEBlock(ConvSODEFunc((1, 28, 28), 1, 10)),
            ODEBlock(ConvSODEFunc((10, 28, 28), 10, 10)),
            ODEBlock(ConvSODEFunc((10, 28, 28), 10, 10))
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            ODEBlock(ConvSODEFunc((10, 14, 14), 10, 20)),
            ODEBlock(ConvSODEFunc((20, 14, 14), 20, 20)),
            ODEBlock(ConvSODEFunc((20, 14, 14), 20, 20))
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            ODEBlock(ConvSODEFunc((20, 7, 7), 20, 40)),
            ODEBlock(ConvSODEFunc((40, 7, 7), 40, 40)),
            ODEBlock(ConvSODEFunc((40, 7, 7), 40, 40))
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            ODEBlock(ConvSODEFunc((40, 3, 3), 40, 20)),
            ODEBlock(ConvSODEFunc((20, 3, 3), 20, 20)),
            ODEBlock(ConvSODEFunc((20, 3, 3), 20, 40))
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            ODEBlock(ConvSODEFunc((80, 7, 7), 80, 40)),
            ODEBlock(ConvSODEFunc((40, 7, 7), 40, 20)),
            ODEBlock(ConvSODEFunc((20, 7, 7), 20, 20))
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            ODEBlock(ConvSODEFunc((40, 14, 14), 40, 20)),
            ODEBlock(ConvSODEFunc((20, 14, 14), 20, 10)),
            ODEBlock(ConvSODEFunc((10, 14, 14), 10, 10))
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            ODEBlock(ConvSODEFunc((20, 28, 28), 20, 10)),
            ODEBlock(ConvSODEFunc((10, 28, 28), 10, 10)),
            ODEBlock(ConvSODEFunc((10, 28, 28), 10, 10, normalize=False))
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 14, 14)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
            nn.SiLU()
        )

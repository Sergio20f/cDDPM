from functools import wraps
from packaging import version
from collections import namedtuple
import matplotlib.pyplot as plt
import imageio
import einops
import numpy as np

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
from einops import rearrange


# constants
AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers
def exists(val):
    return val is not None

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class
class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu
        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention
        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash:
            return self.flash_attn(q, k, v)

        scale = q.shape[-1] ** -0.5

        # similarity
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention
        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


MAX_NUM_STEPS = 1000

def sinusoidal_embedding(n, d):
    """
    Generates sinusoidal embeddings for conditioning the model on the current time step.

    Args:
        n (int): The number of time steps.
        d (int): The dimensionality of the embedding.

    Returns:
        torch.Tensor: A tensor of shape (n, d) representing the sinusoidal embeddings.

    """
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

    return embedding


def _make_te(self, dim_in, dim_out):
    """
    Utility function that creates a one-layer MLP used for mapping a positional embedding.

    Args:
        dim_in (int): The input dimensionality of the MLP.
        dim_out (int): The output dimensionality of the MLP.

    Returns:
        torch.nn.Sequential: A one-layer MLP module.

    """
    return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.SiLU(),
        nn.Linear(dim_out, dim_out)
    )


def get_nonlinearity(name):
    """Helper function to get non linearity module, choose from relu/softplus/swish/lrelu"""
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "softplus":
        return nn.Softplus()
    elif name == "swish":
        return Swish(inplace=True)
    elif name == "lrelu":
        return nn.LeakyReLU()


class Swish(nn.Module):
    def __init__(self, inplace=False):
        """The Swish non linearity function"""
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        out = self.attend(q, k, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)

        return self.to_out(out)


class InitialVelocity(nn.Module):
    def __init__(self, nf, non_linearity="relu"):
        super(InitialVelocity, self).__init__()

        self.norm1 = nn.InstanceNorm2d(nf)
        #self.norm1 = nn.InstanceNorm2d(1)
        #self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        # For 1 channel images (?)
        self.conv1 = nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(nf)
        #self.conv2 = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(nf, nf*2, kernel_size=1, stride=1, padding=0)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x0):
        out = self.norm1(x0)
        out = self.conv1(out)
        out = self.non_linearity(out)
        out = self.norm2(out)
        out = self.conv2(out)
        out = self.non_linearity(out)
        return out #torch.cat((x0, out), dim=1)


class ODEBlock(nn.Module):
    def __init__(self, odefunc, tol=1e-3, adjoint=False):
        """
        Code adapted from https://github.com/EmilienDupont/augmented-neural-odes
        Utility class that wraps odeint and odeint_adjoint.

        Wrapper that takes the function f (given by ConvSODEFunc) and solves the
        ODE using that functionimport matplotlib.pyplot as plt
import imageio
import einops
import numpy as np. ConvSODEFunc defines the actual transformation of
        the datam while the ODE block is responsible for solving the ODE defined
        by that transformation.

        Args:
            odefunc (nn.Module): the module to be evaluated
            tol (float): tolerance for the ODE solver
            adjoint (bool): whether to use the adjoint method for gradient calculation
        """
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, time_emb, eval_times=None, method="rk4"):
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        if self.adjoint:
            out = odeint_adjoint(
                lambda t, x: self.odefunc(t, x, time_emb=time_emb),
                x,
                integration_time,
                rtol=self.tol,
                atol=self.tol,
                method=method,
                options={"max_num_steps": MAX_NUM_STEPS},
            )
        else:
            out = odeint(
                lambda t, x: self.odefunc(t, x, time_emb=time_emb),
                x,
                integration_time,
                rtol=self.tol,
                atol=self.tol,
                method=method,
                options={"max_num_steps": MAX_NUM_STEPS},
            )

        if eval_times is None:
            return out[1]  # out[1][:int(len(x)/2)]  Return only final time
        else:
            return out

    def trajectory(self, x, timesteps):
        integration_time = torch.linspace(0.0, 1.0, timesteps)
        return self.forward(x, eval_times=integration_time)


class Conv2dTime(nn.Conv2d):
    def __init__(self, in_channels, *args, **kwargs):
        """
        Code adapted from https://github.com/EmilienDupont/augmented-neural-odes
        Conv2d module where time gets concatenated as a feature map.
        Makes ODE func aware of the current time step.
        """
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        # Shape (batch_size, 1, height, width)
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        # Shape (batch_size, channels + 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)


class ConvSODEFunc(nn.Module):
    def __init__(self, nf, time_dependent=False, non_linearity="relu",
                 time_emb_dim=None):
        """
        Block for ConvSODEUNet. Designed to be used as the function f that defines
        the derivative in an ODE of the form dz/dt = f(t, z). This function represents
        how the hidden state z changes with respect to the continuous time variable t.

        ConvSODE defines the actual transformation of the datam while the ODE
        block is responsible for solving the ODE defined by that transformation.

        Args:
            nf (int): number of filters for the conv layers
            time_dependent (bool): whether to concat the time as a feature map before the convs
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
        """
        super(ConvSODEFunc, self).__init__()

        dim = nf
        dim_out = nf

        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        if time_dependent:
            self.norm1 = nn.InstanceNorm2d(nf)
            self.conv1 = Conv2dTime(nf, nf, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.InstanceNorm2d(nf)
            self.conv2 = Conv2dTime(nf, nf, kernel_size=3, stride=1, padding=1)
        else:
            self.norm1 = nn.InstanceNorm2d(nf)
            self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.InstanceNorm2d(nf)
            # changed to kernel_size 1 with padding 0 instead of 1 - WHY?
            # Working with kernel size 1 and padding 0
            self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        # 1x1 convolution used for the residual connection
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, t, x, time_emb=None):
        scale_shift = None
        cutoff = int(x.shape[1] / 2)  # int(len(x)/2)
        # Typical in neural ODEs as they usually operate in anaugmented state space
        z = x[:, :cutoff]
        v = x[:, cutoff:]
        into = torch.cat((z, v), dim=1)
        self.nfe += 1

        if self.time_dependent:
            out = self.norm1(into)
            out = self.conv1(t, into)
            out = self.non_linearity(out)
            out = self.norm2(out)

            if exists(self.mlp) and exists(time_emb):
              time_emb = self.mlp(time_emb)
              time_emb = rearrange(time_emb, 'b c -> b c 1 1')
              scale_shift = time_emb.chunk(2, dim=1)
              scale, shift = scale_shift
              out = out * (scale + 1) + shift

            out = self.conv2(t, out)
            out = self.non_linearity(out)
        else:
            out = self.norm1(into)
            out = self.conv1(out)
            out = self.non_linearity(out)
            out = self.norm2(out)

            if exists(self.mlp) and exists(time_emb):
              time_emb = self.mlp(time_emb)
              time_emb = rearrange(time_emb, 'b c -> b c 1 1')
              scale_shift = time_emb.chunk(2, dim=1)
              scale, shift = scale_shift
              out = out * (scale + 1) + shift

            out = self.conv2(out)
            out = self.non_linearity(out)

        return out + self.res_conv(x)


# Second-order ODE UNet
class ConvSODEUNet(nn.Module):
    def __init__(
        self,
        n_steps=1000,
        time_emb_dim=256, # num_filters*16
        channels=3, # change to 3
        num_filters=16,
        out_dim=3, # change to 3
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

        #################################################################
        self.channels = channels
        self.out_dim = out_dim
        self.random_or_learned_sinusoidal_cond = None
        self.self_condition = False
        #################################################################

        #################################################################
        self.attention_encoder4 = Attention(dim=nf*16)

        self.attention_decoder1 = Attention(dim=nf*16)
        #################################################################

        #################################################################
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        #################################################################
        self.initial_velocity = InitialVelocity(nf, non_linearity)

        ode_down1 = ConvSODEFunc(nf * 2, time_dependent, non_linearity) # 32
        self.odeblock_down1 = ODEBlock(ode_down1, tol=tol, adjoint=adjoint)
        self.conv_down1_2 = nn.Conv2d(nf * 2, nf * 4, 1, 1)

        ode_down2 = ConvSODEFunc(nf * 4, time_dependent, non_linearity) # 64
        self.odeblock_down2 = ODEBlock(ode_down2, tol=tol, adjoint=adjoint)
        self.conv_down2_3 = nn.Conv2d(nf * 4, nf * 8, 1, 1)

        ode_down3 = ConvSODEFunc(nf * 8, time_dependent, non_linearity) # 128
        self.odeblock_down3 = ODEBlock(ode_down3, tol=tol, adjoint=adjoint)
        self.conv_down3_4 = nn.Conv2d(nf * 8, nf * 16, 1, 1)

        ode_down4 = ConvSODEFunc(nf * 16, time_dependent, non_linearity) # 256
        self.odeblock_down4 = ODEBlock(ode_down4, tol=tol, adjoint=adjoint)
        self.conv_down4_embed = nn.Conv2d(nf * 16, nf * 32, 1, 1)

        ode_embed = ConvSODEFunc(nf * 32, time_dependent, non_linearity) # 512
        self.odeblock_embedding = ODEBlock(ode_embed, tol=tol, adjoint=adjoint)
        self.conv_up_embed_1 = nn.Conv2d(nf * 32 + nf * 16, nf * 16, 1, 1)

        ode_up1 = ConvSODEFunc(nf * 16, time_dependent, non_linearity) # 256
        self.odeblock_up1 = ODEBlock(ode_up1, tol=tol, adjoint=adjoint)
        self.conv_up1_2 = nn.Conv2d(nf * 16 + nf * 8, nf * 8, 1, 1)

        ode_up2 = ConvSODEFunc(nf * 8, time_dependent, non_linearity) # 128
        self.odeblock_up2 = ODEBlock(ode_up2, tol=tol, adjoint=adjoint)
        self.conv_up2_3 = nn.Conv2d(nf * 8 + nf * 4, nf * 4, 1, 1)

        ode_up3 = ConvSODEFunc(nf * 4, time_dependent, non_linearity) # 64
        self.odeblock_up3 = ODEBlock(ode_up3, tol=tol, adjoint=adjoint)
        self.conv_up3_4 = nn.Conv2d(nf * 4 + nf * 2, nf * 2, 1, 1)

        ode_up4 = ConvSODEFunc(nf * 2, time_dependent, non_linearity) # 32
        self.odeblock_up4 = ODEBlock(ode_up4, tol=tol, adjoint=adjoint)

        self.classifier = nn.Conv2d(nf * 2, out_dim, 1)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x, t, dummy=None):
      t = self.time_embed(t)
      x = self.initial_velocity(x)

      features1 = self.odeblock_down1(x, method=self.method, time_emb=t)
      x = self.non_linearity(self.conv_down1_2(features1))
      x = nn.functional.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)

      features2 = self.odeblock_down2(x, method=self.method, time_emb=t)
      x = self.non_linearity(self.conv_down2_3(features2))
      x = nn.functional.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)

      features3 = self.odeblock_down3(x, method=self.method, time_emb=t)
      x = self.non_linearity(self.conv_down3_4(features3))
      x = nn.functional.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)

      features4 = self.odeblock_down4(x, method=self.method, time_emb=t)
      features4 = self.attention_encoder4(features4) # Attention in encoder
      x = self.non_linearity(self.conv_down4_embed(features4))
      x = nn.functional.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)

      x = self.odeblock_embedding(x, method=self.method, time_emb=t)

      x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
      x = torch.cat((x, features4), dim=1)
      x = self.non_linearity(self.conv_up_embed_1(x))
      x = self.odeblock_up1(x,  method=self.method, time_emb=t)
      x = self.attention_decoder1(x)  # Attention in decoder

      x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
      x = torch.cat((x, features3), dim=1)
      x = self.non_linearity(self.conv_up1_2(x))
      x = self.odeblock_up2(x,  method=self.method, time_emb=t)

      x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
      x = torch.cat((x, features2), dim=1)
      x = self.non_linearity(self.conv_up2_3(x))
      x = self.odeblock_up3(x,  method=self.method, time_emb=t)

      x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
      x = torch.cat((x, features1), dim=1)
      x = self.non_linearity(self.conv_up3_4(x))
      x = self.odeblock_up4(x,  method=self.method, time_emb=t)

      pred = self.classifier(x)
      return pred


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

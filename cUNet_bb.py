import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint

from utils import get_nonlinearity


MAX_NUM_STEPS = 1000


class InitialVelocity(nn.Module):
    def __init__(self, nf, non_linearity="relu"):
        super(InitialVelocity, self).__init__()

        #self.norm1 = nn.InstanceNorm2d(nf)
        self.norm1 = nn.InstanceNorm2d(1)
        #self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        # For 1 channel images (?)
        self.conv1 = nn.Conv2d(1, nf, kernel_size=3, stride=1, padding=1)
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
        Args:
            odefunc (nn.Module): the module to be evaluated
            tol (float): tolerance for the ODE solver
            adjoint (bool): whether to use the adjoint method for gradient calculation
        """
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None, method="rk4"):
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        if self.adjoint:
            out = odeint_adjoint(
                self.odefunc,
                x,
                integration_time,
                rtol=self.tol,
                atol=self.tol,
                method=method,
                options={"max_num_steps": MAX_NUM_STEPS},
            )
        else:
            out = odeint(
                self.odefunc,
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
    def __init__(self, nf, time_dependent=False, non_linearity="relu"):
        """
        Block for ConvSODEUNet
        Args:
            nf (int): number of filters for the conv layers
            time_dependent (bool): whether to concat the time as a feature map before the convs
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
        """
        super(ConvSODEFunc, self).__init__()
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations

        if time_dependent:
            self.norm1 = nn.InstanceNorm2d(nf)
            self.conv1 = Conv2dTime(nf, nf, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.InstanceNorm2d(nf)
            self.conv2 = Conv2dTime(nf, nf, kernel_size=3, stride=1, padding=1)
        else:
            self.norm1 = nn.InstanceNorm2d(nf)
            self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.InstanceNorm2d(nf)
            # changed to kernel_size 1 with padding 0 instead of 1
            self.conv2 = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, t, x):
        cutoff = int(x.shape[1] / 2)  # int(len(x)/2)
        z = x[:, :cutoff]
        v = x[:, cutoff:]
        into = torch.cat((z, v), dim=1)
        self.nfe += 1
        if self.time_dependent:
            out = self.norm1(into)
            out = self.conv1(t, into)
            out = self.non_linearity(out)
            out = self.norm2(out)
            out = self.conv2(t, out)
            out = self.non_linearity(out)
        else:
            out = self.norm1(into)
            out = self.conv1(out)
            out = self.non_linearity(out)
            out = self.norm2(out)
            out = self.conv2(out)
            out = self.non_linearity(out)
        return out
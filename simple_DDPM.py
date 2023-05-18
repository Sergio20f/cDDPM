import torch
import torch.nn as nn


class MyDDPM(nn.Module):
    """
    MyDDPM (My Differentiable Diffusion Probabilistic Model) class for modeling and manipulating data.

    Args:
        network (nn.Module): The neural network model.
        n_steps (int): The number of diffusion steps.
        min_beta (float): The minimum value of beta.
        max_beta (float): The maximum value of beta.
        device (torch.device): The device to use for computation.
        image_chw (tuple): The shape of the input image (channels, height, width).

    Attributes:
        n_steps (int): The number of diffusion steps.
        device (torch.device): The device used for computation.
        image_chw (tuple): The shape of the input image (channels, height, width).
        network (nn.Module): The neural network model.
        betas (torch.Tensor): The sequence of beta values.
        alphas (torch.Tensor): The sequence of alpha values (1 - betas).
        alpha_bars (torch.Tensor): The cumulative product of alphas.

    """

    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 28, 28)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        """
        Forward pass of the MyDDPM model.

        Args:
            x0 (torch.Tensor): The input image tensor.
            t (int): The diffusion step index.
            eta (torch.Tensor): The noise tensor. If None, it will be randomly generated.

        Returns:
            torch.Tensor: The noisy image tensor.

        """
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t):
        """
        Backward pass of the MyDDPM model.

        Args:
            x (torch.Tensor): The input image tensor.
            t (int): The diffusion step index.

        Returns:
            torch.Tensor: The estimated noise tensor.

        """
        return self.network(x, t)

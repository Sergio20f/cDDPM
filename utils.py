import matplotlib.pyplot as plt
import imageio
import einops
import numpy as np
import torch

import torch.nn as nn


def show_images(images, title=""):
    """Displays a grid of images as sub-plots in a square layout.

    Args:
        images (torch.Tensor or numpy.ndarray): Input images to be displayed. If `images` is a tensor, it will be
            converted to a CPU numpy array.
        title (str, optional): Title of the figure. Default is an empty string.

    Raises:
        TypeError: If `images` is not a torch.Tensor or numpy.ndarray.

    """
    # Converting images to CPU numpy arrays
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    elif not isinstance(images, np.ndarray):
        raise TypeError("Input 'images' must be a torch.Tensor or numpy.ndarray.")

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()


# Shows the first batch of images
def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], "Images in the first batch")
        break


def show_forward(ddpm, loader, device):
    """Displays the forward process of a Denoising Diffusion Probabilistic Model (DDPM) on a given image loader.

    Args:
        ddpm (torch.nn.Module): The DDPM model to visualize.
        loader (torch.utils.data.DataLoader): DataLoader containing the images to process.
        device (torch.device): Device (CPU or GPU) on which to perform the forward process.

    """
    # Showing the forward process
    for batch in loader:
        imgs = batch[0]

        show_images(imgs, "Original images")

        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(
                ddpm(imgs.to(device), [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]),
                f"DDPM Noisy images {int(percent * 100)}%"
            )
        break


def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=100, gif_name="sampling.gif", c=1, h=28, w=28):
    """Generates new samples using a Denoising Diffusion Probabilistic Model (DDPM).

    Args:
        ddpm (torch.nn.Module): The DDPM model used for generating new samples.
        n_samples (int, optional): The number of samples to generate. Defaults to 16.
        device (torch.device, optional): The device (CPU or GPU) on which to perform the generation. If not specified, the device of the DDPM model will be used.
        frames_per_gif (int, optional): The number of frames per GIF to create. Defaults to 100.
        gif_name (str, optional): The name of the GIF file to save. Defaults to "sampling.gif".
        c (int, optional): The number of channels in the generated samples. Defaults to 1.
        h (int, optional): The height of the generated samples. Defaults to 28.
        w (int, optional): The width of the generated samples. Defaults to 28.

    Returns:
        torch.Tensor: The generated samples.

    """
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)

        # Looping through the diffusion steps
        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                # Rendering frame
                frames.append(frame)

    # Storing the GIF
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)
            if idx == len(frames) - 1:
                for _ in range(frames_per_gif // 3):
                    writer.append_data(frames[-1])
                    
    return x


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

import torch
import torch.nn as nn
from tqdm.auto import tqdm
import json

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST

from utils import show_images, generate_new_images
from simple_DDPM import MyDDPM
from UNet import MyUNet


def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            # Loading data
            x0 = batch[0].to(device)
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Read the config file
    with open('config.json', 'r') as file:
        config = json.load(file)

    # Access the parameter values
    STORE_PATH_MNIST = config['STORE_PATH_MNIST']
    STORE_PATH_FASHION = config['STORE_PATH_FASHION']
    no_train = config['no_train']
    fashion = config['fashion']
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    lr = config['lr']
    store_path = config['store_path']

    # Loading the data (converting each image into a tensor and normalizing between [-1, 1])
    transform = Compose([
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2)]
    )
    ds_fn = FashionMNIST if fashion else MNIST
    dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    # Defining model
    n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors (include in config)
    ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

    # Training
    store_path = "ddpm_fashion.pt" if fashion else "ddpm_mnist.pt"
    if not no_train:
        training_loop(ddpm, loader, n_epochs, optim=Adam(ddpm.parameters(), lr), device=device, store_path=store_path)
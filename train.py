from denoising_diffusion_pytorch import GaussianDiffusion, Trainer
from cUNet import ConvSODEUNet


model = ConvSODEUNet(n_steps=1000)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 256    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    #"trainingSet/trainingSet/",
    #"PetImages/Cat",
    #"cifar10/train",
    #"celeba_hq_256",
    "img_align_celeba",
    train_batch_size = 128,
    train_lr =2e-4, #8e-5,
    train_num_steps = 500000, #700000, # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.9999, # 0.995       # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False             # whether to calculate fid during training
)

# Load the desired model
#milestone = 10  # Specify the milestone or identifier of the saved model
#trainer.load(milestone)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in cUNet: {total_params}")

print("Training Starts")
trainer.train()

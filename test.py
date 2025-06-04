from pipeline_dmd import DMDPipeline
from diffusers import UNet2DModel, DDIMScheduler
import os
import torch
from tqdm import tqdm

unet = UNet2DModel.from_pretrained("dmd", subfolder="generator")

scheduler = DDIMScheduler.from_pretrained("configs")

pipe = DMDPipeline(unet, scheduler)

pipe.to(torch.device("mps:0"))

save_directory = "images"

os.makedirs(save_directory, exist_ok=True)

for i in tqdm(range(16)):
    generator = torch.manual_seed(i)
    image = pipe(generator=generator).images[0]
    image.save(os.path.join(save_directory, f"image-{i + 1}.png"))


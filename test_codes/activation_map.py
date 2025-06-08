#%%
import numpy as np
import PIL
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler
)

#%%
torch.cuda.set_device(5)
# Load main pipeline
model_id = "finetuned_models/ip2p_nollm_res256_lr5e-5_pretrained_unet_1000steps_13laststeps"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

#%%
# Image preparation
generator = torch.Generator("cuda").manual_seed(0)
image_path = "test_imgs/org.jpg"

def load_image(img_path):
    image = PIL.Image.open(img_path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

image = load_image(image_path)

#%%
plt.imshow(np.array(image))
plt.axis("off")
plt.title("Original Image")
plt.show()

#%%
# Setup hook storage
activations = {}

def get_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations[name] = output.detach().cpu()
    return hook

# Register hooks for down blocks, mid block, up blocks
for i, block in enumerate(pipe.unet.down_blocks):
    block.register_forward_hook(get_hook(f"down_blocks_{i}"))

pipe.unet.mid_block.register_forward_hook(get_hook("mid_block"))

for i, block in enumerate(pipe.unet.up_blocks):
    block.register_forward_hook(get_hook(f"up_blocks_{i}"))

    
#%%
# Inference
prompt = "The girl's hands were not flat on the bench"
num_inference_steps = 10
image_guidance_scale = 1.5
guidance_scale = 10

edited_image = pipe(
    prompt,
    image=image,
    num_inference_steps=num_inference_steps,
    image_guidance_scale=image_guidance_scale,
    guidance_scale=guidance_scale,
    generator=generator,
).images[0]

# Show result
plt.imshow(edited_image)
plt.axis("off")
plt.title("Edited Image")
plt.show()

#%%
def show_activation_separate(activation, title, max_channels=8):
    act = activation
    if act.dim() == 4:
        act = act[0]
    n_channels = min(max_channels, act.shape[0])

    plt.figure(figsize=(2*n_channels, 2))
    for i in range(n_channels):
        plt.subplot(1, n_channels, i+1)
        img = act[i].cpu().numpy()
        plt.imshow(img, cmap='viridis')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


#%%
# Show activations for down_blocks, mid_block, up_blocks
for name, activation in activations.items():
    show_activation_separate(activation, title=f"{name}", max_channels=8)

# %%

#%%
import numpy as np
import PIL
import os
import sys
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler
)
import torch.nn.functional as F

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
finetuned_models_path = os.path.join(parent_dir, 'finetuned_models')
if finetuned_models_path not in sys.path:
    sys.path.insert(0, finetuned_models_path)
    
#%%
torch.cuda.set_device(6)
# Model directory name (relative to finetuned_models)
model_id = "ip2p_nollm_res256_lr5e-5_pretrained_unet_1000steps_13laststeps"

# Full path to the model directory
model_path = os.path.join(finetuned_models_path, model_id)

# Load main pipeline from the correct folder
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

#%%
# Image preparation
generator = torch.Generator("cuda").manual_seed(0)
image_path = "image_org.jpg"

def load_image(img_path):
    image = PIL.Image.open(img_path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

image = load_image(image_path)

# %%
activations = {}
activation_grads = {}

def get_hook_with_grad(name):
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations[name] = output
        def grad_hook(grad):
            activation_grads[name] = grad
        output.register_hook(grad_hook)
    return forward_hook

# %%
# Register new hooks for saliency
for i, block in enumerate(pipe.unet.down_blocks):
    block.register_forward_hook(get_hook_with_grad(f"down_blocks_{i}"))
pipe.unet.mid_block.register_forward_hook(get_hook_with_grad("mid_block"))
for i, block in enumerate(pipe.unet.up_blocks):
    block.register_forward_hook(get_hook_with_grad(f"up_blocks_{i}"))

# Prepare image for input (requires_grad not needed for PIL input)
# If your pipeline requires a torch tensor input, set requires_grad=True

# Forward pass with torch.no_grad() OFF for gradients
pipe.enable_model_cpu_offload()  # if needed for memory
pipe.unet.eval()  # Ensure in eval mode
for p in pipe.unet.parameters():
    p.requires_grad_(True)
    
# %%
# Inference
prompt = "Let the red panda raise its paws"
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

# Choose a simple scalar output for saliency (mean pixel value)
output_tensor = torch.tensor(np.array(edited_image), dtype=torch.float32).mean()
output_tensor.backward()

# %%

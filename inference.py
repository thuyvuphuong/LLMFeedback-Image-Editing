#%%
import PIL
import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler
)
import matplotlib.pyplot as plt

#%%
# Load main pipeline
model_id = "pretrained_frameworks/pretrained_IEDMs/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

#%%
# Override safety checker and scheduler
# pipe.safety_checker = lambda images, clip_input: (images, [False])
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Load UNet from a different path
# new_unet_path = "pretrained_frameworks/pretrained_IEDMs/magicbrush-jul7/unet"
# pipe.unet = UNet2DConditionModel.from_pretrained(new_unet_path, torch_dtype=torch.float16).to("cuda")

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
# Inference
prompt = "The girl's hands were not flat on the bench"
num_inference_steps = 20
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

plt.imshow(edited_image)
# edited_image.save("edited_image1.png")

# %%

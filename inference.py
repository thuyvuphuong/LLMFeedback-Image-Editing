#%%
import PIL
import torch
import matplotlib.pyplot as plt
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
pipe.unet = UNet2DConditionModel.from_pretrained("finetuned_models/ip2p_llm_start0.9_des0.5_den0.5_res256_lr5e-4_pretrained_unet_1000steps_4nextepochs/checkpoint-20/unet", torch_dtype=torch.float16).to("cuda")

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

plt.imshow(edited_image)
# edited_image.save("edited_image1.png")

# %%

#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Diffusion for InstructPix2Pix."""

import argparse
import logging
import math
import os
import sys
import shutil
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm
import accelerate
import datasets
import numpy as np
import PIL
import json
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from transformers import AutoModelForCausalLM
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from activation_saliency_utils import get_saliency_masks_of_batch, register_hooks

output_dir = "test_output"
os.makedirs(output_dir, exist_ok=True)

#import libraries for SceneGraph
from glob import glob
sys.path.append(os.path.abspath('./pretrained_frameworks/SceneGraph/egtr'))
from model.deformable_detr import DeformableDetrConfig
from model.egtr import DetrForSceneGraphGeneration
from test_codes.get_scenegraph import inference_one_image_get_scenegraph_only

#import libraries for LLM
sys.path.append(os.path.abspath('./pretrained_frameworks/LLMs/DeepSeek-VL2'))
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

model_path = "./pretrained_frameworks/LLMs/DeepSeek-VL2/pretrained_models/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer_llm = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.float16).cuda().eval()

embedding_sentence_model = SentenceTransformer('pretrained_frameworks/all-MiniLM-L6-v2')

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__, log_level="INFO")

#Declaration of global variables for SceneGraph
data_path = "./pretrained_frameworks/SceneGraph/egtr/data/visual_genome"
artifact_path = "./pretrained_frameworks/SceneGraph/egtr/pretrained/egtr_vg_pretrained"
architecture = "./pretrained_frameworks/SceneGraph/egtr/pretrained/deformable-detr"
logit_adjustment = False
logit_adj_tau = 0.3
min_size = 800
max_size = 1333
max_topk = 100

cfg_path = './pretrained_frameworks/DepthEstimation/Depth-Anything/checkpoints/config_vitl14.json'
pth_path = './pretrained_frameworks/DepthEstimation/Depth-Anything/checkpoints/depth_anything_vitl14.pth'

with open(cfg_path) as f:
    cfg = json.load(f)
weights = torch.load(pth_path)

config = DeformableDetrConfig.from_pretrained(artifact_path)
config.logit_adjustment = logit_adjustment
config.logit_adj_tau = logit_adj_tau

scengraph_model = DetrForSceneGraphGeneration.from_pretrained(
    architecture, config=config, ignore_mismatched_sizes=True,
)
ckpt_path = sorted(
    glob(f"{artifact_path}/checkpoints/epoch=*.ckpt"),
    key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
)[-1]
state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
for k in list(state_dict.keys()):
    state_dict[k[6:]] = state_dict.pop(k)  # "model."

scengraph_model.load_state_dict(state_dict)
scengraph_model.cuda()

#For Stable Diffusion

DATASET_NAME_MAPPING = {
    "downloaded_datatset/HumanEdit": ("IMAGE_ID", "EDITING_TYPE", "CORE", "MASK",
                                      "EDITING_INSTRUCTION", "OUTPUT_DESCRIPTION", 
                                      "INPUT_CAPTION_BY_LLAMA", "OUTPUT_CAPTION_BY_LLAMA", 
                                      "INPUT_IMG", "MASK_IMG", "OUTPUT_IMG"),
}
WANDB_TABLE_COL_NAMES = ["INPUT_IMG", "OUTPUT_IMG", "EDITING_INSTRUCTION"]


def log_validation(
    pipeline,
    args,
    accelerator,
    generator,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    original_image = download_image(args.val_image_url)
    edited_images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        for _ in range(args.num_validation_images):
            edited_images.append(
                pipeline(
                    args.validation_prompt,
                    image=original_image,
                    num_inference_steps=20,
                    image_guidance_scale=1.5,
                    guidance_scale=7,
                    generator=generator,
                ).images[0]
            )

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
            for edited_image in edited_images:
                wandb_table.add_data(wandb.Image(original_image), wandb.Image(edited_image), args.validation_prompt)
            tracker.log({"validation": wandb_table})

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_unet_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained UNet or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_LLM_feedback",
        type=bool,
        default=False,
        help="Whether to use LLM feedback or not.",
    )
    parser.add_argument(
        "--use_localize_loss",
        type=bool,
        default=False,
        help="Whether to use localization loss or not.",
    )
    parser.add_argument(
        "--LLM_start_ratio",
        type=float,
        default=0.33,
        help="Ratio of total training steps after which LLM feedback is applied (0.0 to 1.0).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--input_ids_column",
        type=str,
        default="IMAGE_ID",
        help="The column of the dataset containing the image IDs.",
    )
    parser.add_argument(
        "--original_image_column",
        type=str,
        default="INPUT_IMG",
        help="The column of the dataset containing the original image on which edits where made.",
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default="OUTPUT_IMG",
        help="The column of the dataset containing the edited image.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default="EDITING_INSTRUCTION",
        help="The column of the dataset containing the edit instruction.",
    )
    parser.add_argument(
        "--editing_mask_column",
        type=str,
        default="MASK_IMG",
        help="The column of the dataset containing the edit mask.",
    )
    parser.add_argument(
        "--target_prompt_column",
        type=str,
        default=None,
        help="The column of the dataset containing the target prompt.",
    )
    parser.add_argument(
        "--val_image_url",
        type=str,
        default=None,
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=None,
        help=(
            "Threshold to get binary mask, range: 0-1"
        ),
    )
    parser.add_argument(
        "--timestep_threshold",
        type=int,
        default=None,
        help=(
            "Timestep threshold"
        ),
    )
    parser.add_argument(
        "--exclude_layers_index_list",
        type=list_of_ints,
        default=None,
        help=(
            "List of indices of layers excluded from the mean. indices=0,1,2,3,4,5,6,7,8"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="instruct-pix2pix-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def convert_org_mask_to_binary(org_mask, resolution):
    mask = org_mask.convert("RGB").resize((resolution, resolution))
    gray_img = mask.convert("L")
    binary_img = gray_img.point(lambda x: 255 if x == 0 else 0, '1')
    return np.array(binary_img)


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True, timeout=DIFFUSERS_REQUEST_TIMEOUT).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def soft_iou_loss(pred, target, eps=1e-6):
    # Both pred and target are floats in [0, 1], shape: (B, H, W)
    intersection = (pred * target).sum(dim=(1, 2))
    union = (pred + target - pred * target).sum(dim=(1, 2))
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou.mean()


def compute_batch_iou(mask1, mask2):
    # Ensure masks are boolean
    mask1 = mask1.bool()
    mask2 = mask2.bool()

    # Compute intersection and union
    intersection = (mask1 & mask2).float().sum(dim=(1, 2))
    union = (mask1 | mask2).float().sum(dim=(1, 2))

    # Compute IoU with numerical stability
    iou = intersection / (union + 1e-8)

    # Clamp values to [0, 1]
    iou = iou.clamp(0, 1)
    return iou


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_unet_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )
    
    layers_to_hook = [
        "down_blocks.0",
        "down_blocks.1",
        "down_blocks.2",
        "down_blocks.3",
        "mid_block",
        "up_blocks.0",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3"
    ]

    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
    in_channels = 8
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)

    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :8, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/main/en/image_load#imagefolder
        
    def replace_with_float_index(example, idx):
        example["IMAGE_ID"] = float(idx)
        return example

    dataset["train"] = dataset["train"].map(replace_with_float_index, with_indices=True)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.input_ids_column is None:
        input_ids_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        input_ids_column = args.input_ids_column
        if input_ids_column not in column_names:
            raise ValueError(
                f"--input_ids_column' value '{args.input_ids_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.original_image_column is None:
        original_image_column = dataset_columns[8] if dataset_columns is not None else column_names[8]
    else:
        original_image_column = args.original_image_column
        if original_image_column not in column_names:
            raise ValueError(
                f"--original_image_column' value '{args.original_image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edit_prompt_column is None:
        edit_prompt_column = dataset_columns[4] if dataset_columns is not None else column_names[4]
    else:
        edit_prompt_column = args.edit_prompt_column
        if edit_prompt_column not in column_names:
            raise ValueError(
                f"--edit_prompt_column' value '{args.edit_prompt_column}' needs to be one of: {', '.join(column_names)}"
            )
            
    if args.edited_image_column is None:
        edited_image_column = dataset_columns[10] if dataset_columns is not None else column_names[10]
    else:
        edited_image_column = args.edited_image_column
        if edited_image_column not in column_names:
            raise ValueError(
                f"--edited_image_column' value '{args.edited_image_column}' needs to be one of: {', '.join(column_names)}"
            )
            
    if args.editing_mask_column is None:
        editing_mask_column = dataset_columns[9] if dataset_columns is not None else column_names[9]
    else:
        editing_mask_column = args.editing_mask_column
        if editing_mask_column not in column_names: 
            raise ValueError(
                f"--editing_mask_column' value '{args.editing_mask_column}' needs to be one of: {', '.join(column_names)}"
            )
    
    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    def preprocess_images(examples):
        original_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[original_image_column]]
        )
        edited_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[edited_image_column]]
        )
        mask_values = np.stack(
            [convert_org_mask_to_binary(mask, args.resolution) for mask in examples[editing_mask_column]]
        )
        mask_values = torch.tensor(mask_values, dtype=torch.int8)
        
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = np.stack([original_images, edited_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return train_transforms(images), mask_values
    

    def preprocess_train(examples):
        # Preprocess images.
        preprocessed_images, preprocessed_masks = preprocess_images(examples)

        original_images, edited_images = preprocessed_images
        original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)

        # Collate the preprocessed images into the `examples`.
        examples["original_pixel_values"] = original_images
        examples["edited_pixel_values"] = edited_images
        examples["mask_values"] = preprocessed_masks

        # Preprocess the captions.
        edit_promtps = list(examples[edit_prompt_column])
        input_ids = list(examples[input_ids_column])
        
        examples["edit_prompts"] = tokenize_captions(edit_promtps)
        examples["input_ids"] = input_ids
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
        edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
        mask_pixel_values = torch.stack([example["mask_values"] for example in examples])
        mask_pixel_values = mask_pixel_values.to(memory_format=torch.contiguous_format).int()
        edit_prompts = torch.stack([example["edit_prompts"] for example in examples])
        input_ids = torch.tensor([example["input_ids"] for example in examples])
        return {
            "input_ids": input_ids,
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "mask_pixel_values": mask_pixel_values,
            "edit_prompts": edit_prompts,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes
    llm_start_step = int(args.LLM_start_ratio * num_training_steps_for_scheduler)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )
    
    hook_handles = register_hooks(unet, layers_to_hook)
    
    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    def sanitize_config(config):
        return {k: (str(v) if not isinstance(v, (int, float, str, bool, torch.Tensor)) else v)
                for k, v in config.items()}
    config = sanitize_config(vars(args))
    
    if accelerator.is_main_process:
        accelerator.init_trackers("finetune-semantic", config=config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  LLM feedback start = {llm_start_step}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        iou_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            images = batch["original_pixel_values"]
            target_images = batch["edited_pixel_values"]
            mask_label = batch["mask_pixel_values"]
            
            org_batch = ((images+1)/2 * 255).to(torch.uint8)
            org_batch = org_batch.permute(0, 2, 3, 1).cpu().numpy()
            pils_batch = [Image.fromarray(image).convert("RGB") for image in org_batch]
            
            tar_batch = ((target_images+1)/2 * 255).to(torch.uint8)
            tar_batch = tar_batch.permute(0, 2, 3, 1).cpu().numpy()
            tar_pils_batch = [Image.fromarray(tar_image).convert("RGB") for tar_image in tar_batch]
            
            input_ids_batch = batch["input_ids"]
            
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning.
                encoder_hidden_states = text_encoder(batch["edit_prompts"])[0]

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                    # Sample masks for the original images.
                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    original_image_embeds = image_mask * original_image_embeds

                # Concatenate the `original_image_embeds` with the `noisy_latents`.
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)
                concatenated_noisy_latents.requires_grad_()

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                
                denoising_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                if not args.use_localize_loss and args.use_LLM_feedback and global_step >= llm_start_step:
                    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=unwrap_model(unet),
                        text_encoder=unwrap_model(text_encoder),
                        vae=unwrap_model(vae),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.safety_checker = lambda images, clip_input: (images, [False])
                    pipeline.set_progress_bar_config(disable=True)
                    
                    if torch.backends.mps.is_available():
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast(accelerator.device.type)

                    with autocast_ctx:
                        temp_desc_loss = 0
                        for i in tqdm(range(len(pils_batch))):
                            input_image = pils_batch[i]
                            target_image = tar_pils_batch[i]
                            index = int(input_ids_batch[i])
                            edit_prompt = dataset["train"][index][args.edit_prompt_column]
                            # target_prompt = dataset["train"][index][args.target_prompt_column]
                            
                            image = pipeline(
                                prompt=edit_prompt,
                                image=input_image,
                                num_inference_steps=20,
                                image_guidance_scale=1.5,
                                guidance_scale=10,
                                num_images_per_prompt=1,
                            ).images[0]
                            pil_img = image.convert("RGB")
                            pil_img_list_form = [pil_img]
                            tar_img_list_form = [target_image]
                            
                            #get scene graph
                            # scenegraph_output = inference_one_image_get_scenegraph_only(scengraph_model, pil_img)
                            # print(scenegraph_output)
                            
                            #create LLM prompt
                            # llm_prompt = "Based on the given image <image>\n, Analyze the image in extreme detail and describe everything visible in a single, coherent paragraph. Should describe mostly based on the image. Additional information also should be used as reference: " + scenegraph_output
                            llm_prompt = f'''<image> \n
                                            You are a meticulous visual analyst. Carefully examine the given image and describe it in a single, flowing paragraph (maximum 520 tokens). Focus on every visually observable detail—such as color, texture, material, size, shape, and spatial relationships. Do not use bullet points or lists.
                                            Avoid assumptions or inferences about unseen factors (e.g., time of day, season, emotions, story). Describe only what is directly visible in the image.
                                            Your paragraph must naturally include the following:
                                            - A clear overview of the setting (e.g., indoor/outdoor, environment type, lighting conditions, background elements, overall mood)
                                            - Detailed description of each major object: its appearance, color, material (wood, metal, fabric, etc.), texture (smooth, rough, shiny, soft, etc.), size (relative to others), and spatial position (e.g., foreground, center-left)
                                            - If humans or animals are present, describe each individual separately in full detail. Include:
                                                - Hair, face, visible skin or fur, and accessories
                                                - Clothing (color, texture, material, style, condition)
                                                - Pose: the orientation and position of every visible body part (head, arms, legs, torso, hands, feet)
                                                - Describe their stance or motion only if clearly visible, grounded in what is seen
                                            - For images with multiple people or animals, ensure each is described distinctly and thoroughly, woven into the flow of the paragraph
                                            - Describe all supporting/background elements such as furniture, walls, ground, vegetation, or objects in the distance
                                            - Clearly express spatial relationships between elements (e.g., in front of, behind, next to, overlapping, under)
                                            - You must explicitly describe the visual features of each object or region targeted in the editing instruction: "{edit_prompt}", separately. For example, if the instruction is "The girl bent and raised her two hands," then describe: The girl posture (e.g., leaning forward, bent knees) and The position and gesture of her hands (e.g., raised above shoulders, palms open)
                                            Use vivid, sensory-rich language. Every detail must be grounded in what can actually be seen. Avoid summarizing—immerse the reader in a scene constructed entirely from the image visible content.
                                            '''
                            
                            conversation = [
                                {
                                    "role": "<|User|>",
                                    "content": llm_prompt,
                                },
                                {"role": "<|Assistant|>", "content": ""},
                            ]
                                
                            prepare_inputs = vl_chat_processor(
                                conversations=conversation,
                                images=pil_img_list_form,
                                force_batchify=True,
                                system_prompt=""
                            ).to(vl_gpt.device)
                            
                            prepare_inputs_tar = vl_chat_processor(
                                conversations=conversation,
                                images=tar_img_list_form,
                                force_batchify=True,
                                system_prompt=""
                            ).to(vl_gpt.device)
                            
                            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                            outputs = vl_gpt.language.generate(
                                inputs_embeds=inputs_embeds,
                                attention_mask=prepare_inputs.attention_mask,
                                pad_token_id=tokenizer_llm.eos_token_id,
                                bos_token_id=tokenizer_llm.bos_token_id,
                                eos_token_id=tokenizer_llm.eos_token_id,
                                max_new_tokens=256,
                                do_sample=False,
                                use_cache=True
                            )
                            image_description = [tokenizer_llm.decode(output.cpu().tolist(), skip_special_tokens=True) for output in outputs]
                            
                            inputs_embeds_tar = vl_gpt.prepare_inputs_embeds(**prepare_inputs_tar)
                            tar_outputs = vl_gpt.language.generate(
                                inputs_embeds=inputs_embeds_tar,
                                attention_mask=prepare_inputs_tar.attention_mask,
                                pad_token_id=tokenizer_llm.eos_token_id,
                                bos_token_id=tokenizer_llm.bos_token_id,
                                eos_token_id=tokenizer_llm.eos_token_id,
                                max_new_tokens=256,
                                do_sample=False,
                                use_cache=True
                            )
                            target_image_description = [tokenizer_llm.decode(tar_output.cpu().tolist(), skip_special_tokens=True) for tar_output in tar_outputs]
                            
                            # print(image_description)
                            # print(target_image_description)
                            # print("---------------------------")
                            
                            
                            embeddings1 = embedding_sentence_model.encode(target_image_description, convert_to_tensor=True, show_progress_bar=False)
                            embeddings2 = embedding_sentence_model.encode(image_description, convert_to_tensor=True, show_progress_bar=False)

                            similarity = util.cos_sim(embeddings1, embeddings2).item()
                            description_loss_per_one = 1 - similarity
                            # print(description_loss_per_one)
                            temp_desc_loss += description_loss_per_one
                            
                        desc_loss = temp_desc_loss / len(pils_batch)
                        loss = 0.5 * desc_loss + 0.5 * denoising_loss
                        
                # elif args.use_localize_loss and not args.use_LLM_feedback:
                #     loss = 0.5 * denoising_loss + 0.5 * iou_loss
                #     print("Training with localization loss", loss)
                    
                # elif not args.use_localize_loss and not args.use_LLM_feedback:
                #     loss = denoising_loss
                else: 
                    loss = denoising_loss                           
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                
                resolution = (args.resolution, args.resolution)
                saliency_masks = get_saliency_masks_of_batch(
                    resolutions=resolution,
                    exclude_indices=args.exclude_layers_index_list,
                    mask_threshold=args.mask_threshold,
                    binary_mask=True
                )
                saliency_mask_tensor = torch.tensor(saliency_masks, dtype=torch.int8, device=accelerator.device)
                timestep_mask = (timesteps < args.timestep_threshold).to(torch.int)
                timestep_mask_expanded = timestep_mask[:, None, None]
                
                saliency_mask_filtered_with_time = saliency_mask_tensor * timestep_mask_expanded
                mask_label_filtered_with_time = mask_label * timestep_mask_expanded
                
                iou_loss = soft_iou_loss(saliency_mask_filtered_with_time, mask_label_filtered_with_time)
                
                # saliency_np = saliency_mask_tensor[0].cpu().numpy().astype(np.uint8) * 255
                # Image.fromarray(saliency_np).save(os.path.join(output_dir, f"{iou_loss}_saliency_mask.jpg"))

                # mask_label_np = mask_label[0].cpu().numpy().astype(np.uint8) * 255
                # Image.fromarray(mask_label_np).save(os.path.join(output_dir, f"{iou_loss}_mask_label.jpg"))
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0 or global_step == num_training_steps_for_scheduler-1:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": denoising_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break         

        if accelerator.is_main_process:
            if (
                (args.val_image_url is not None)
                and (args.validation_prompt is not None)
                and (epoch % args.validation_epochs == 0)
            ):
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                # The models need unwrapping because for compatibility in distributed training mode.
                pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    text_encoder=unwrap_model(text_encoder),
                    vae=unwrap_model(vae),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )

                log_validation(
                    pipeline,
                    args,
                    accelerator,
                    generator,
                )

                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

                del pipeline
                torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=unwrap_model(text_encoder),
            vae=unwrap_model(vae),
            unet=unwrap_model(unet),
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        if (args.val_image_url is not None) and (args.validation_prompt is not None):
            log_validation(
                pipeline,
                args,
                accelerator,
                generator,
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()
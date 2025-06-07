import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import cv2
import json
from glob import glob
sys.path.append(os.path.abspath('./pretrained_frameworks/SceneGraph/egtr'))
sys.path.append(os.path.abspath('./pretrained_frameworks/DepthEstimation/Depth-Anything'))

from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from model.egtr import DetrForSceneGraphGeneration
from train_egtr import collate_fn, evaluate_batch
from data.visual_genome import VGDataset
from util.box_ops import rescale_bboxes
from lib.pytorch_misc import argsort_desc

from torchvision.transforms import Compose
import torch.nn.functional as F
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from depth_anything.dpt import DepthAnything

from depth_process import get_depths, filter_depth_and_replace_with_mean

data_path = "./pretrained_frameworks/SceneGraph/egtr/data/visual_genome"
artifact_path = "./pretrained_frameworks/SceneGraph/egtr/pretrained/egtr_vg_pretrained"
architecture = "./pretrained_frameworks/SceneGraph/egtr/pretrained/deformable-detr"
num_queries = 200
split = "val"
batch_size = 4
logit_adjustment = False
logit_adj_tau = 0.3
min_size = 800
max_size = 1333
num_workers = 4
max_topk = 100

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg_path = './pretrained_frameworks/DepthEstimation/Depth-Anything/checkpoints/config_vitl14.json'
pth_path = './pretrained_frameworks/DepthEstimation/Depth-Anything/checkpoints/depth_anything_vitl14.pth'

with open(cfg_path) as f:
    cfg = json.load(f)
weights = torch.load(pth_path)

# depth_anything = DepthAnything(cfg).to(DEVICE).eval()
# depth_anything.load_state_dict(weights)
# total_params = sum(param.numel() for param in depth_anything.parameters())
# print('Total parameters in depth estimation model: {:.2f}M'.format(total_params / 1e6))

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

# Feature extractor
feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
    architecture, size=min_size, max_size=max_size
)

dataset = VGDataset(
    data_folder=data_path,
    feature_extractor=feature_extractor,
    split=split,
    num_object_queries=num_queries,
)
dataloader = DataLoader(
    dataset,
    collate_fn=lambda x: collate_fn(x, feature_extractor, transform),
    batch_size=batch_size,
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True,
)

rel_categories = ["__background__", "above", "across", "against", "along", "and", "at", 
                      "attached to", "behind", "belonging to", "between", "carrying", "covered in", 
                      "covering", "eating", "flying in", "for", "from", "growing on", 
                      "hanging from", "has", "holding", "in", "in front of", "laying on", 
                      "looking at", "lying on", "made of", "mounted on", "near", "of", "on", 
                      "on back of", "over", "painted on", "parked on", "part of", "playing", 
                      "riding", "says", "sitting on", "standing on", "to", "under", "using", 
                      "walking in", "walking on", "watching", "wearing", "wears", "with"]
id2label = {
        k - 1: v["name"] for k, v in dataset.coco.cats.items()
    }
num_labels = max(id2label.keys()) + 1
output_file_name = "scenegraph_output.json"

@torch.no_grad()
def inference_batch(model, dataloader, dataset, max_topk=max_topk,
                    rel_categories=rel_categories, save=False, save_path = None):
    output_file = os.path.join(save_path, output_file_name)
    id2label = {
            k - 1: v["name"] for k, v in dataset.coco.cats.items()
        }
    num_labels = max(id2label.keys()) + 1
    model.eval()
    
    results = []
    for batch in tqdm(dataloader):
        raw_imgs = batch["raw_imgs"]
        raw_sizes = batch["raw_sizes"]
        outputs = model(
            pixel_values=batch["pixel_values"].cuda(),
            pixel_mask=batch["pixel_mask"].cuda(),
            output_attentions=False,
            output_attention_states=True,
            output_hidden_states=True,
        )
        for j in range(outputs["logits"].size(0)):  # Iterate over batch size
            #calculate objects depth
            # image = raw_imgs[j].to(DEVICE)
            # h, w = raw_sizes[j]
            # depth = depth_anything(image)
            # depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            # depth = depth.cpu().numpy().astype(np.uint8)
            # depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            
            img_id = batch["labels"][j]["image_id"]
            img_name = f"{int(img_id)}.jpg"
            orig_size = batch["labels"][j]["orig_size"]
            pred_logits = outputs["logits"][j]
            obj_scores, pred_classes = torch.max(
                pred_logits.softmax(-1)[:, :num_labels], -1
            )
            sub_ob_scores = torch.outer(obj_scores, obj_scores)
            sub_ob_scores[
                torch.arange(pred_logits.size(0)), torch.arange(pred_logits.size(0))
            ] = 0.0
            
            pred_boxes = outputs["pred_boxes"][j]
            pred_boxes = rescale_bboxes(pred_boxes.cpu(), torch.flip(orig_size, dims=[0]))
            # objects_depth_mean = get_depths(depth, pred_boxes)
            # filtered_mean_depth = filter_depth_and_replace_with_mean(objects_depth_mean, pred_classes)
            
            pred_rel = torch.clamp(outputs["pred_rel"][j], 0.0, 1.0)
            if "pred_connectivity" in outputs:
                pred_connectivity = torch.clamp(outputs["pred_connectivity"][j], 0.0, 1.0)
                pred_rel = torch.mul(pred_rel, pred_connectivity)
                
            triplet_scores = torch.mul(pred_rel, sub_ob_scores.unsqueeze(-1))
            pred_rel_inds = argsort_desc(triplet_scores.cpu().clone().numpy())[
                :max_topk, :
            ]  # [pred_rels, 3(s,o,p)]
            rel_scores = (
                pred_rel.cpu()
                .clone()
                .numpy()[pred_rel_inds[:, 0], pred_rel_inds[:, 1], pred_rel_inds[:, 2]]
            )
            
            pred_entry = {
                "pred_boxes": pred_boxes.clone().numpy(),
                "pred_classes": pred_classes.cpu().clone().numpy(),
                "obj_scores": obj_scores.cpu().clone().numpy(),
                "pred_rel_inds": pred_rel_inds,
                "rel_scores": rel_scores,
            }

            scenegraph_out = ""
            for idx, (sub_idx, obj_idx, rel_idx) in enumerate(pred_entry["pred_rel_inds"]):
                sub_idx = int(sub_idx)
                obj_idx = int(obj_idx)
                
                sub_label = id2label.get(pred_entry["pred_classes"][sub_idx], f"label_{pred_entry['pred_classes'][sub_idx]}")
                obj_label = id2label.get(pred_entry["pred_classes"][obj_idx], f"label_{pred_entry['pred_classes'][obj_idx]}")
                rel_label = rel_categories[rel_idx + 1]  # +1 because index 0 is __background__
                score = pred_entry["rel_scores"][idx]
                scenegraph_out = scenegraph_out +  f"{sub_label} {rel_label} {obj_label}, "
            if save: 
                output_data = {
                    "file_names": img_name,
                    "relation": scenegraph_out
                }
                results.append(output_data)
                
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=4)
                # print(
                #     f"{sub_label}[{filtered_mean_depth[sub_idx]}] {rel_label} {obj_label}[{filtered_mean_depth[obj_idx]}]"
                # )

    print(f"Results saved to {output_file}")

    
# Model
config = DeformableDetrConfig.from_pretrained(artifact_path)
config.logit_adjustment = logit_adjustment
config.logit_adj_tau = logit_adj_tau

model = DetrForSceneGraphGeneration.from_pretrained(
    architecture, config=config, ignore_mismatched_sizes=True
)
ckpt_path = sorted(
    glob(f"{artifact_path}/checkpoints/epoch=*.ckpt"),
    key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
)[-1]
state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
for k in list(state_dict.keys()):
    state_dict[k[6:]] = state_dict.pop(k)  # "model."

model.load_state_dict(state_dict)
model.cuda()

inference_batch(model, dataloader, dataset, save=True, save_path ="text_docs")
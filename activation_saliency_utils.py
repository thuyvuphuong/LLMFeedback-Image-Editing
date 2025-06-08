import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np
from torchvision.utils import save_image

# Store activations in a global dictionary
ACTIVATIONS = {}

def get_module_by_name(model, module_name):
    # Helper to get module by its dotted name
    names = module_name.split(".")
    m = model
    for n in names:
        m = getattr(m, n)
    return m

def save_activation(name):
    def hook(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        act.retain_grad()
        ACTIVATIONS[name] = act
    return hook

def register_hooks(model, layers_to_hook):
    handles = []
    for name in layers_to_hook:
        m = get_module_by_name(model, name)
        handles.append(m.register_forward_hook(save_activation(name)))
    return handles

def compute_saliency_map(input_image, model, target_layer, target_idx=None):
    """
    Args:
        input_image: torch.Tensor of shape (1, C, H, W), requires_grad=True
        model: the model (e.g., UNet)
        target_layer: name of the layer to register hook
        target_idx: if not None, index in the output to compute gradients for
    Returns:
        saliency: grad of output w.r.t input
        activation: activation from target_layer
    """
    ACTIVATIONS.clear()
    handles = register_hooks(model, [target_layer])
    input_image.requires_grad_()
    output = model(input_image)
    if target_idx is not None:
        score = output.flatten()[target_idx]
    else:
        score = output.mean()
    model.zero_grad()
    score.backward()
    saliency = input_image.grad.detach().cpu()
    activation = ACTIVATIONS[target_layer]
    for h in handles:
        h.remove()
    return saliency, activation

import os
import numpy as np
import cv2

def overlay_saliency_on_image(
    ACTIVATIONS,
    input_image,
    folder_name="sample_1",
    root_folder="all_samples"
):
    """
    Saves input image and saliency overlays (one per block) to a unique folder inside root_folder.
    - input_image can be a PIL Image or numpy array [H,W,3], uint8.
    - Each block's saliency overlay is saved as {block}_saliency_overlay.png.
    - The input image is saved as input.png.
    - root_folder/folder_name/ is created if it does not exist.
    """
    output_dir = os.path.join(root_folder, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Convert PIL Image to numpy if needed
    if hasattr(input_image, 'size'):  # PIL Image
        input_image = np.array(input_image)
    # If grayscale, convert to 3-channel
    if input_image.ndim == 2:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    elif input_image.shape[2] == 4:  # RGBA to BGR
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2BGR)

    H_img, W_img = input_image.shape[:2]

    # Save input image
    input_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, "input.png"), input_bgr)

    for name, act in ACTIVATIONS.items():
        if act.grad is None:
            continue
        grad_mean = act.grad.mean(dim=1, keepdim=True).cpu()
        grad_map = grad_mean[0, 0].detach().cpu().numpy()  # [H, W]

        # Defensive checks
        if grad_map.size == 0 or np.isnan(grad_map).any() or np.isinf(grad_map).any():
            print(f"Warning: grad_map for {name} is invalid, skipping.")
            continue

        # Normalize to [0, 1]
        grad_map = grad_map - np.min(grad_map)
        if np.max(grad_map) != 0:
            grad_map = grad_map / np.max(grad_map)
        else:
            grad_map = np.zeros_like(grad_map)

        # Resize to input_image size
        grad_map_resized = cv2.resize(grad_map.astype(np.float32), (W_img, H_img), interpolation=cv2.INTER_CUBIC)

        # Apply Jet colormap
        grad_map_colored = cv2.applyColorMap(np.uint8(255 * grad_map_resized), cv2.COLORMAP_JET)
        # Overlay: blend with input image
        overlay = cv2.addWeighted(input_image, 0.6, grad_map_colored, 0.4, 0)
        # Save overlay
        cv2.imwrite(os.path.join(output_dir, f"{name}_saliency_overlay.png"), overlay)        

def save_first_sample_mean_images(ACTIVATIONS, concatenated_noisy_latents, save_dir="output_maps_cv2"):
    """
    For each activation in ACTIVATIONS:
      - Compute mean over channels, shape: [B, 1, H, W]
      - Take the first sample [1, H, W]
      - Repeat along channel to [C, H, W], where C=concatenated_noisy_latents.shape[1]
      - For each channel, normalize to [0,255] and save as PNG using cv2
    Same for saliency (grad) if present.
    """
    os.makedirs(save_dir, exist_ok=True)
    C = concatenated_noisy_latents.shape[1]

    for name, act in ACTIVATIONS.items():
        # Activation mean: [B, 1, H, W]
        act_mean = act.mean(dim=1, keepdim=True).cpu()
        first_act = act_mean[0]  # [1, H, W]
        first_act_repeated = first_act.repeat(C, 1, 1)  # [C, H, W]
        # Save each channel as a grayscale image
        for ch in range(C):
            arr = first_act_repeated[ch].detach().cpu().numpy()
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)  # Normalize to [0,1]
            arr = (arr * 255).astype(np.uint8)
            filename = os.path.join(save_dir, f"{name}_activation_ch{ch}.png")
            cv2.imwrite(filename, arr)

        # Saliency mean (if grad exists)
        if act.grad is not None:
            grad_mean = act.grad.mean(dim=1, keepdim=True).cpu()
            first_grad = grad_mean[0]
            first_grad_repeated = first_grad.repeat(C, 1, 1)
            for ch in range(C):
                arr = first_grad_repeated[ch].numpy()
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                arr = (arr * 255).astype(np.uint8)
                filename = os.path.join(save_dir, f"{name}_saliency_ch{ch}.png")
                cv2.imwrite(filename, arr)
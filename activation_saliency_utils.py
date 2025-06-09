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

def get_saliency_masks_of_batch(
    resolutions,
    exclude_indices=None,
    mask_threshold=0.5,
    binary_mask=True
):
    """
    Returns a list of binary masks (one per image in batch) from the mean saliency map,
    thresholded at mask_threshold.
    - ACTIVATIONS: dict of {name: activation} as usual, but each activation has a batch dim.
    - resolutions: [(H, W), ...] list of output mask shapes for each image, or a single (H, W) for all.
    - exclude_indices: list/set/tuple of indices to skip in mean computation (default None)
    """

    # Determine batch size from any activation
    sample_act = next(iter(ACTIVATIONS.values()))
    batch_size = sample_act.grad.shape[0]

    # Handle single resolution for all images
    if isinstance(resolutions, tuple):
        resolutions = [resolutions] * batch_size

    masks = []
    for b in range(batch_size):
        saliency_maps = []
        for name, act in ACTIVATIONS.items():
            if not hasattr(act, "grad") or act.grad is None:
                continue
            grad_mean = act.grad.mean(dim=1, keepdim=True).cpu()
            grad_map = grad_mean[b, 0].detach().cpu().numpy()

            if grad_map.size == 0 or np.isnan(grad_map).any() or np.isinf(grad_map).any():
                continue

            grad_map = grad_map - np.min(grad_map)
            max_grad = np.max(grad_map)
            if max_grad != 0:
                grad_map = grad_map / max_grad
            else:
                grad_map = np.zeros_like(grad_map)

            H_img, W_img = resolutions[b]
            grad_map_resized = cv2.resize(grad_map.astype(np.float32), (W_img, H_img), interpolation=cv2.INTER_CUBIC)
            saliency_maps.append(grad_map_resized)

        mask = None
        if saliency_maps:
            indices = list(range(len(saliency_maps)))
            if exclude_indices is not None:
                indices = [i for i in indices if i not in set(exclude_indices)]
            if not indices:
                raise ValueError("All saliency maps were excluded from mean computation.")
            filtered_maps = [saliency_maps[i] for i in indices]

            mean_saliency = np.mean(np.stack(filtered_maps), axis=0)
            max_val = np.max(mean_saliency)
            threshold = mask_threshold * max_val
            if binary_mask == True:
                mask = (mean_saliency >= threshold).astype(np.uint8)
            else:
                mask = (mean_saliency >= threshold).astype(np.uint8) * 255
            

        masks.append(mask)
    ACTIVATIONS.clear()
    return np.array(masks)


def get_first_saliency_mask_of_batch(
    ACTIVATIONS,
    input_image,
    folder_name=None,
    root_folder=None,
    exclude_indices=None,
    save=False,
    mask_threshold=0.5,
):
    """
    Saves input image and saliency overlays (one per block) to a unique folder inside root_folder.
    Additionally, saves the mean saliency map overlay and its binary mask.
    Optionally, exclude certain block indices from the mean saliency computation.
    - input_image can be a PIL Image or numpy array [H,W,3], uint8.
    - Each block's saliency overlay is saved as {block}_saliency_overlay.png.
    - The input image is saved as input.png.
    - The mean saliency overlay is saved as mean_saliency_overlay.png.
    - The binary mask is saved as mean_saliency_mask.png.
    - root_folder/folder_name/ is created if it does not exist.
    - exclude_indices: list/set/tuple of indices to skip in mean computation (default None)
    """
    # Check save mode and path validity
    if save:
        if root_folder is None or folder_name is None:
            raise ValueError("Saving is enabled, but root_folder and folder_name must be provided.")
        output_dir = os.path.join(root_folder, folder_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None  # Not used if not saving

    # Convert PIL Image to numpy if needed
    if hasattr(input_image, 'size'):  # PIL Image
        input_image = np.array(input_image)
    # If grayscale, convert to 3-channel
    if input_image.ndim == 2:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    elif input_image.shape[2] == 4:  # RGBA to BGR
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2BGR)

    if input_image.dtype != np.uint8:
        input_image = np.clip(input_image, 0, 255).astype(np.uint8)

    H_img, W_img = input_image.shape[:2]

    # Save input image
    if save:
        input_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, "input.png"), input_bgr)

    saliency_maps = []
    names = []
    for name, act in ACTIVATIONS.items():
        if not hasattr(act, "grad") or act.grad is None:
            continue
        grad_mean = act.grad.mean(dim=1, keepdim=True).cpu()
        grad_map = grad_mean[0, 0].detach().cpu().numpy()

        if grad_map.size == 0 or np.isnan(grad_map).any() or np.isinf(grad_map).any():
            print(f"Warning: grad_map for {name} is invalid, skipping.")
            continue

        grad_map = grad_map - np.min(grad_map)
        max_grad = np.max(grad_map)
        if max_grad != 0:
            grad_map = grad_map / max_grad
        else:
            grad_map = np.zeros_like(grad_map)

        grad_map_resized = cv2.resize(grad_map.astype(np.float32), (W_img, H_img), interpolation=cv2.INTER_CUBIC)
        saliency_maps.append(grad_map_resized)
        names.append(name)

        grad_map_colored = cv2.applyColorMap(np.uint8(255 * grad_map_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR), 0.6, grad_map_colored, 0.4, 0)
        if save:
            cv2.imwrite(os.path.join(output_dir, f"{name}_saliency_overlay.png"), overlay)

    mask = None
    if saliency_maps:
        indices = list(range(len(saliency_maps)))
        if exclude_indices is not None:
            indices = [i for i in indices if i not in set(exclude_indices)]
        if not indices:
            raise ValueError("All saliency maps were excluded from mean computation.")
        filtered_maps = [saliency_maps[i] for i in indices]

        mean_saliency = np.mean(np.stack(filtered_maps), axis=0)
        mean_saliency_vis = mean_saliency - np.min(mean_saliency)
        max_mean = np.max(mean_saliency_vis)
        if max_mean != 0:
            mean_saliency_vis = mean_saliency_vis / max_mean
        else:
            mean_saliency_vis = np.zeros_like(mean_saliency)

        mean_saliency_colored = cv2.applyColorMap(np.uint8(255 * mean_saliency_vis), cv2.COLORMAP_JET)
        mean_overlay = cv2.addWeighted(cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR), 0.6, mean_saliency_colored, 0.4, 0)
        if save:
            cv2.imwrite(os.path.join(output_dir, "mean_saliency_overlay.png"), mean_overlay)

        max_val = np.max(mean_saliency)
        threshold = mask_threshold * max_val
        mask = (mean_saliency >= threshold).astype(np.uint8) * 255
        if save:
            mask_path = os.path.join(output_dir, "mean_saliency_mask.png")   
            cv2.imwrite(mask_path, mask)
    return mask

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
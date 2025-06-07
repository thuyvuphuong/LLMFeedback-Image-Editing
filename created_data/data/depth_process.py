import numpy as np

def get_depth(depth_patch, bbox):
    threshold = 0.5 * depth_patch.max()
    depth_prediction_mask = (depth_patch > threshold).astype(np.float32)
    depth_path_with_mask = depth_patch * depth_prediction_mask
    depth_values = depth_path_with_mask[depth_path_with_mask != 0]
    return round(float(np.mean(depth_values)), 2)

def filter_depth_and_replace_with_mean(mean_depths_arr, classes_tensor):
    """
    Filters mean depths by grouping indices of identical class values and calculating 
    the mean for the corresponding mean depths. Then replaces all values in mean_depths_arr 
    with the calculated mean of their respective class group.

    Args:
        mean_depths_arr (numpy.ndarray): Array containing mean depths.
        classes_tensor (torch.Tensor): Tensor containing class values.

    Returns:
        numpy.ndarray: Updated mean_depths_arr with values replaced by the mean of their respective class group.
    """
    # Convert classes_tensor to a NumPy array
    classes_arr = classes_tensor.cpu().clone().numpy()

    # Find unique classes
    unique_classes = np.unique(classes_arr)

    # Create a copy of mean_depths_arr to update
    updated_mean_depths_arr = np.copy(mean_depths_arr)

    for cls in unique_classes:
        # Find indices of the current class
        indices = np.where(classes_arr == cls)[0]

        # Compute mean of mean_depths_arr for these indices
        mean_depth = np.mean(mean_depths_arr[indices])

        # Replace values at these indices with the computed mean
        updated_mean_depths_arr[indices] = mean_depth

    return updated_mean_depths_arr
    

def get_depths(depth_image, bboxes):
    depth_mean_in_boxes = []
    
    for bbox in bboxes:
        # Extract bbox coordinates
        x_topleft = int(bbox[0])
        y_topleft = int(bbox[1])
        width = int(bbox[2])
        height = int(bbox[3])
        
        # Calculate bottom-right coordinates
        x_min = x_topleft
        x_max = x_topleft + width
        y_min = y_topleft
        y_max = y_topleft + height
        
        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        x_max = min(depth_image.shape[1], x_max)
        y_min = max(0, y_min)
        y_max = min(depth_image.shape[0], y_max)
        
        # Extract depth patch
        depth_patch = depth_image[y_min:y_max, x_min:x_max]
        
        # Compute depth mean in the bounding box
        if depth_patch.size > 0:
            depth_mean = np.mean(depth_patch)
        else:
            depth_mean = 0  # Default value if the patch is empty
        
        depth_mean_in_boxes.append(int(depth_mean))
    
    return np.array(depth_mean_in_boxes)
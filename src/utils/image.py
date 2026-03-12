import numpy as np

def normalize_for_detection(img_raw, p_low=1, p_high=99):
    """
    Normalizes a 16-bit image to 8-bit using percentile stretching.
    Uses downsampling for speed.
    Returns: 8-bit RGB/BGR image.
    """
    if img_raw is None: return None
    if img_raw.dtype == np.uint8: return img_raw

    stride = 10 # Downsample for statistics calculation
    
    p_low_vals, p_high_vals = [], []
    if len(img_raw.shape) == 3:
        for i in range(img_raw.shape[2]):
            channel_data = img_raw[::stride, ::stride, i]
            if channel_data.max() > 0:
                p_low_vals.append(np.percentile(channel_data, p_low))
                p_high_vals.append(np.percentile(channel_data, p_high))
    else:
        channel_data = img_raw[::stride, ::stride]
        if channel_data.max() > 0:
            p_low_vals.append(np.percentile(channel_data, p_low))
            p_high_vals.append(np.percentile(channel_data, p_high))

    if not p_low_vals: return np.zeros_like(img_raw, dtype=np.uint8)
        
    vmin = min(p_low_vals)
    vmax = max(p_high_vals)
    
    if vmax <= vmin: return np.zeros_like(img_raw, dtype=np.uint8)
        
    img_f = img_raw.astype(np.float32)
    img_norm = (np.clip(img_f, vmin, vmax) - vmin) / (vmax - vmin) * 255.0
    
    return img_norm.astype(np.uint8)
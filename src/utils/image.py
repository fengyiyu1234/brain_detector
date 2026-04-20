import numpy as np

# def normalize_for_detection(img_raw, p_low=1, p_high=99):
#     """
#     Normalizes a 16-bit image to 8-bit using percentile stretching.
#     Uses downsampling for speed.
#     Returns: 8-bit RGB/BGR image.
#     """
#     if img_raw is None: return None
#     if img_raw.dtype == np.uint8: return img_raw

#     stride = 10 # Downsample for statistics calculation
    
#     p_low_vals, p_high_vals = [], []
#     if len(img_raw.shape) == 3:
#         for i in range(img_raw.shape[2]):
#             channel_data = img_raw[::stride, ::stride, i]
#             if channel_data.max() > 0:
#                 p_low_vals.append(np.percentile(channel_data, p_low))
#                 p_high_vals.append(np.percentile(channel_data, p_high))
#     else:
#         channel_data = img_raw[::stride, ::stride]
#         if channel_data.max() > 0:
#             p_low_vals.append(np.percentile(channel_data, p_low))
#             p_high_vals.append(np.percentile(channel_data, p_high))

#     if not p_low_vals: return np.zeros_like(img_raw, dtype=np.uint8)
        
#     vmin = min(p_low_vals)
#     vmax = max(p_high_vals)
    
#     if vmax <= vmin: return np.zeros_like(img_raw, dtype=np.uint8)
        
#     img_f = img_raw.astype(np.float32)
#     img_norm = (np.clip(img_f, vmin, vmax) - vmin) / (vmax - vmin) * 255.0
    
#     return img_norm.astype(np.uint8)

# 找到你的 normalize_for_detection 函数，修改里面求最大最小值的逻辑：

def normalize_for_detection(img_raw, p_low, p_high):
    # 🌟 [新增] 神奇的切片魔法：每隔 4 个像素抽样一次
    # 将 400 万像素浓缩成 25 万像素去估算分位数，速度提升 16 倍，但结果几乎完全一样！
    img_sample = img_raw[::4, ::4] 
    
    # 用降采样后的图去算分位数
    vmin = np.percentile(img_sample, p_low)
    vmax = np.percentile(img_sample, p_high)
    
    # ... 后续的拉伸和截断逻辑保持你原来的不变，但应用于原图 img_raw ...
    # 比如：
    img_norm = np.clip(img_raw, vmin, vmax)
    img_norm = (img_norm - vmin) / (vmax - vmin + 1e-8) * 255.0
    return img_norm.astype(np.uint8)
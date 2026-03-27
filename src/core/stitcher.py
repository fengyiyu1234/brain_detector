import numpy as np

def stitchDetection(detections, H, W, xsize=512, ysize=512, step=448):
    x_overlap = xsize-step; y_overlap = ysize-step
    rows = []
    for row in range(step,W,step): rows.extend(list(range(row-32,row+x_overlap+32)))
    cols = []
    for col in range(step,H,step): cols.extend(list(range(col-32,col+y_overlap+32)))
    overlap_idx = []
    for i, detection in enumerate(list(detections)):
        box = list(map(int, detection[:-1]))
        if (box[0] in rows) or (box[1] in cols): overlap_idx.append(i)
    overlap_detections = detections[overlap_idx].copy()
    clean_mask = np.ones(detections.shape[0], dtype=bool)
    clean_mask[overlap_idx] = False
    clean_detections = detections[clean_mask].copy()
    if overlap_detections.size > 1:
        overlap_detections = non_max_suppression_merge(overlap_detections)
        clean_detections = np.append(clean_detections, overlap_detections, axis=0)
    return clean_detections  

def non_max_suppression_merge(boxes, overlapThresh=0.5, sort=4):
    if len(boxes) == 0: return []
    pick = []
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:,sort])
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick]

def combine_predictions(all_predictions, csv_reader, classes, z_start, Z, pos, disp_mat, size, metadata_registry, tile_name, tILESIZE = 2048, file_z0 = None):
    row, col = pos
    ABS_X, ABS_Y, ABS_Z = disp_mat[pos]
    H, W = size
    mask = np.zeros((H,W), dtype=float)
    if col > 0: 
        x_pre_start = disp_mat[row,col-1][0]; y_pre_start = disp_mat[row,col-1][1]       
        mask[max(ABS_Y,y_pre_start):min(ABS_Y+tILESIZE,y_pre_start+tILESIZE),
             max(ABS_X,x_pre_start):min(ABS_X+tILESIZE,x_pre_start+tILESIZE)] = 1
    if row > 0:
        x_pre_start = disp_mat[row-1,col][0]; y_pre_start = disp_mat[row-1,col][1]       
        mask[max(ABS_Y,y_pre_start):min(ABS_Y+tILESIZE,y_pre_start+tILESIZE),
             max(ABS_X,x_pre_start):min(ABS_X+tILESIZE,x_pre_start+tILESIZE)] = 1       
    z0 = z_start - ABS_Z
    z1 = z0 + Z
    
    for row in csv_reader:
        slice_name, x1, y1, x2, y2, class_name, score, mean, z = row[:9]
        z = int(float(z))
        x1 = float(x1); x2 = float(x2); y1 = float(y1); y2 = float(y2)
        score = float(score); mean = float(mean)
        if z-1 in range(z0,z1):
            x1 += ABS_X; x2 += ABS_X; y1 += ABS_Y; y2 += ABS_Y; z = z - z0
            if not mask[int((y1+y2)//2),int((x1+x2)//2)] > 0:
                cell_type_index = 0 if classes.index(class_name) < 3 else 1
                all_predictions[z-1][cell_type_index] = np.concatenate((all_predictions[z-1][cell_type_index],
                                                                         [[x1,y1,x2,y2,score,mean,classes.index(class_name),z]]))
                # 将该细胞的全局质心与名字注册到内存中
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                metadata_registry.append([cx, cy, z, tile_name, slice_name])
    return all_predictions

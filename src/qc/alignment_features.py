"""Frozen T70 landmark definitions, independent of the alignment solver.

Only the bright-region point (last feature) is scored. Extraction rules are
preserved from the October 15 experiment for reproducible coordinates.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

def box_center(cell):
    boxes = np.array(list(cell['per_z_boxes'].values()))
    return np.array([(boxes[:, 0].min()+boxes[:, 2].max())/2,
                     (boxes[:, 1].min()+boxes[:, 3].max())/2,
                     (min(cell['per_z_boxes'])+max(cell['per_z_boxes']))/2])


def eligible(cell, offset, region, fit, guard, radii, size):
    """Whole observed track stays in ROI for all shifts and outside fit interval."""
    shift = np.array([offset[k] for k in ('dx', 'dy', 'dz')])
    for z, box in cell['per_z_boxes'].items():
        if not (box[0] >= 0 and box[1] >= 0 and box[2] <= size and box[3] <= size):
            return False
        if fit[0]-guard <= z < fit[1]+guard:
            return False
        x1, y1, x2, y2 = np.asarray(box) + shift[[0, 1, 0, 1]]
        az = z + shift[2]
        if not (region[0] <= x1-radii[0] and x2+radii[0] <= region[1]
                and region[2] <= y1-radii[1] and y2+radii[1] <= region[3]
                and region[4] <= az-radii[2] and az+radii[2] < region[5]):
            return False
    return True


def fixed_pairs(somas, nuclei, soma_offset, nuclear_offset):
    """Unique baseline containment graph, no nearest-center optimization.
    Nuclear box center lies in an observed soma per-slice box; reject ambiguity.
    """
    edges = []
    for ni, nucleus in enumerate(nuclei):
        point = box_center(nucleus) + nuclear_offset - soma_offset
        for si, soma in enumerate(somas):
            if any(abs(z-point[2]) <= .5 and b[0] <= point[0] < b[2]
                   and b[1] <= point[1] < b[3] for z, b in soma['per_z_boxes'].items()):
                edges.append((si, ni))
    sc = np.bincount([s for s, _ in edges], minlength=len(somas))
    nc = np.bincount([n for _, n in edges], minlength=len(nuclei))
    return [(s, n) for s, n in edges if sc[s] == 1 and nc[n] == 1], len(edges)


def volume_features(volume, mask, origin, nuclear):
    """Largest Otsu component; bright center uses peak's upper-decile component."""
    vals = volume[mask]
    if len(vals) < 8 or not np.isfinite(vals).all() or np.ptp(vals) <= 0:
        return None
    if np.issubdtype(volume.dtype, np.integer) and np.any(vals == np.iinfo(volume.dtype).max):
        return None
    smooth = ndi.gaussian_filter(volume.astype(float), sigma=(0, 1, 1))
    values = smooth[mask]
    signal = np.maximum(smooth-np.percentile(values, 10), 0)
    labels, count = ndi.label(mask & (smooth > threshold_otsu(values)))
    if not count:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = labels == sizes.argmax()
    if component.sum() < 4:
        return None
    weights = signal*component if nuclear else component.astype(float)
    center = np.asarray(ndi.center_of_mass(weights))[::-1] + origin
    labels, _ = ndi.label(component & (signal >= np.percentile(signal[component], 90)))
    peak = np.unravel_index(np.argmax(np.where(component, signal, -np.inf)), signal.shape)
    weights = signal * (labels == labels[peak])
    bright_center = np.asarray(ndi.center_of_mass(weights))[::-1] + origin
    return np.array([center, bright_center])


def extract_features(cell, reader, nuclear):
    boxes = np.array(list(cell['per_z_boxes'].values()))
    x0, y0 = np.maximum(np.floor(boxes[:, :2].min(0)).astype(int)-3, 0)
    first = reader(int(min(cell['per_z_boxes'])))
    x1, y1 = np.minimum(np.ceil(boxes[:, 2:].max(0)).astype(int)+3, first.shape[::-1])
    z0, z1 = int(min(cell['per_z_boxes'])), int(max(cell['per_z_boxes']))
    volume = np.stack([reader(z)[y0:y1, x0:x1] for z in range(z0, z1+1)])
    mask = np.zeros(volume.shape, bool)
    for z, box in cell['per_z_boxes'].items():
        a, b = np.floor(box[:2]).astype(int)
        c, d = np.ceil(box[2:]).astype(int)
        mask[int(z)-z0, b-y0:d-y0, a-x0:c-x0] = True
    points = volume_features(volume, mask, np.array([x0+.5, y0+.5, z0]), nuclear)
    return None if points is None else np.vstack([box_center(cell), points])


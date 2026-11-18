"""Shared review-image helpers for the sample QC checks (read-only).

Images are read from the raw tile folders of each channel. A channel is placed in the
pipeline frame by its Stage 2.5 shift: aligned = raw + (dx, dy, dz), so the aligned
pixel (x, y, z) shows raw (x - dx, y - dy, z - dz). Files smaller than
``min_image_bytes`` are blank planes and are never drawn.

Colours: RFP red, GFP green, Sox9 cyan, Olig2 magenta (any other channel grey/white).
"""
import json
import os
from functools import lru_cache

import numpy as np

from src.utils.io import resolve_tile_dir

REVIEW_DEFAULTS = {
    'enabled': True,
    'min_image_bytes': 1_000_000,
    'max_items_per_check': 5,
    'layers_per_item': 3,
    'crop_px': 512,
}
COLOURS = {'RFP': (1, 0, 0), 'GFP': (0, 1, 0), 'GFPhi': (0, 1, 0),
           'Sox9': (0, 1, 1), 'Olig2': (1, 0, 1)}


def review_options(config):
    out = dict(REVIEW_DEFAULTS)
    out.update(((config.get('qc') or {}).get('review')) or {})
    return out


def _tiffs(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('.'))


class TileImages:
    """Per-channel raw tile images and Stage 2.5 shifts of one sample."""

    def __init__(self, config, results_dir, min_bytes=REVIEW_DEFAULTS['min_image_bytes']):
        self.config, self.results_dir, self.min_bytes = config, results_dir, int(min_bytes)
        paths = config.get('paths') or {}
        self.roots = {}
        for ch in config.get('channels_routing', []):
            if not ch.get('active', True):
                continue
            self.roots[ch['id']] = paths.get(ch.get('dir_key', ''))
            if ch.get('double_exposure'):
                self.roots[ch['second_intensity_id']] = paths.get(ch.get('second_intensity_dir_key', ''))
        self.channels = [ch['id'] for ch in config.get('channels_routing', []) if ch.get('active', True)]
        self.anchor = self.channels[0] if self.channels else None
        self.align_dir = os.path.join(results_dir, '3_2d_aligned')
        self._anchor_dirs = None

    def available(self):
        root = self.roots.get(self.anchor)
        return bool(root) and os.path.isdir(root)

    def anchor_dir(self, tile):
        if self._anchor_dirs is None:
            self._anchor_dirs = {}
            root = self.roots.get(self.anchor)
            if root and os.path.isdir(root):
                for r, d, _ in os.walk(root):
                    if not d:
                        self._anchor_dirs[os.path.basename(r)] = r
        return self._anchor_dirs.get(tile)

    @lru_cache(maxsize=None)
    def files(self, ch, tile):
        anchor = self.anchor_dir(tile)
        root = self.roots.get(ch)
        if not anchor or not root:
            return ()
        rel = os.path.relpath(anchor, self.roots[self.anchor])
        return tuple(_tiffs(resolve_tile_dir(os.path.abspath(root), rel)))

    def file(self, ch, tile, z):
        """1-based slice z of a channel, or None."""
        files = self.files(ch, tile)
        return files[z - 1] if 1 <= z <= len(files) else None

    def is_real(self, ch, tile, z):
        path = self.file(ch, tile, z)
        return bool(path) and os.path.getsize(path) >= self.min_bytes

    def all_real(self, tile, z, channels=None):
        return all(self.is_real(ch, tile, z) for ch in (channels or self.channels))

    @lru_cache(maxsize=None)
    def shift(self, tile, ch):
        path = os.path.join(self.align_dir, f'{tile}_offsets.json')
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as handle:
                value = json.load(handle).get(ch)
            if value:
                return int(value['dx']), int(value['dy']), int(value['dz'])
        return 0, 0, 0

    def read(self, ch, tile, z, window, shift=None):
        """Aligned-frame crop (x0, x1, y0, y1) at aligned slice z; NaN outside/blank."""
        import tifffile
        x0, x1, y0, y1 = window
        dx, dy, dz = self.shift(tile, ch) if shift is None else shift
        out = np.full((y1 - y0, x1 - x0), np.nan, np.float32)
        raw_z = int(round(z - dz))
        path = self.file(ch, tile, raw_z)
        if not path or os.path.getsize(path) < self.min_bytes:
            return out
        img = tifffile.imread(path)
        if img.ndim > 2:
            img = img.reshape(img.shape[-2:]) if img.size == img.shape[-1] * img.shape[-2] else img[0]
        sx, sy = x0 - int(round(dx)), y0 - int(round(dy))
        h, w = img.shape[:2]
        ys, xs, ye, xe = max(0, sy), max(0, sx), min(h, sy + y1 - y0), min(w, sx + x1 - x0)
        if ye > ys and xe > xs:
            out[ys - sy:ye - sy, xs - sx:xe - sx] = img[ys:ye, xs:xe]
        return out


# ── picking ───────────────────────────────────────────────────────────────────

def pick_layers(z_candidates, n=3, ok=None):
    """Up to n slices spread evenly along z among the candidates accepted by ok(z)."""
    zs = sorted({int(z) for z in z_candidates})
    if ok is not None:
        zs = [z for z in zs if ok(z)]
    if len(zs) <= n:
        return zs
    idx = np.linspace(0, len(zs) - 1, n + 2)[1:-1].round().astype(int)
    return [zs[i] for i in idx]


def densest_window(cx, cy, size, tile_size=2048):
    """(x0, x1, y0, y1) of a size x size crop around the densest point cloud area."""
    half = size // 2
    if len(cx) == 0:
        c = tile_size // 2
        return c - half, c + half, c - half, c + half
    bins = max(1, tile_size // 128)
    hist, xe, ye = np.histogram2d(np.asarray(cx), np.asarray(cy), bins=bins,
                                  range=[[0, tile_size], [0, tile_size]])
    i, j = np.unravel_index(int(np.argmax(hist)), hist.shape)
    x = int(np.clip((xe[i] + xe[i + 1]) / 2, half, tile_size - half))
    y = int(np.clip((ye[j] + ye[j + 1]) / 2, half, tile_size - half))
    return x - half, x + half, y - half, y + half


def crop_at(x, y, size, tile_size=2048):
    half = size // 2
    x = int(np.clip(x, half, max(half, tile_size - half)))
    y = int(np.clip(y, half, max(half, tile_size - half)))
    return x - half, x + half, y - half, y + half


# ── drawing ───────────────────────────────────────────────────────────────────

def stretch(img, lo_pct=1, hi_pct=99.5):
    v = img[np.isfinite(img)]
    if not v.size:
        return np.zeros(img.shape, np.float32)
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    return np.nan_to_num(np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)).astype(np.float32)


def composite(images_by_channel):
    """RGB float image from {channel: 2D array} with the fixed channel colours."""
    shape = next(iter(images_by_channel.values())).shape
    rgb = np.zeros(shape + (3,), np.float32)
    for ch, img in images_by_channel.items():
        colour = COLOURS.get(ch, (1, 1, 1))
        n = stretch(img)
        for k in range(3):
            rgb[..., k] += colour[k] * n
    return np.clip(rgb, 0, 1)


def to_pil(img, scale=1):
    from PIL import Image
    arr = img if img.ndim == 3 else np.dstack([stretch(img)] * 3)
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    out = Image.fromarray(arr)
    if scale != 1:
        out = out.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
    return out


def font(size):
    from PIL import ImageFont
    try:
        import matplotlib
        return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans.ttf'), size)
    except Exception:
        return ImageFont.load_default()


def draw_boxes(pil, boxes, window, colour, width=2, scale=1):
    """boxes: iterable of (x1, y1, x2, y2) in the same frame as window."""
    from PIL import ImageDraw
    x0, _, y0, _ = window
    d = ImageDraw.Draw(pil)
    for b in boxes:
        d.rectangle([(b[0] - x0) * scale, (b[1] - y0) * scale, (b[2] - x0) * scale, (b[3] - y0) * scale],
                    outline=colour, width=width)
    return pil


def grid_png(rows, out_png, title='', subtitle='', gap=8):
    """rows: list of rows, each a list of (PIL image, caption). Saved atomically."""
    from PIL import Image, ImageDraw
    head, cap = (54 if subtitle else 30), 22
    widths = [sum(im.size[0] for im, _ in row) + gap * (len(row) - 1) for row in rows]
    heights = [max(im.size[1] for im, _ in row) + cap for row in rows]
    out = Image.new('RGB', (max(widths), head + sum(heights) + gap * (len(rows) - 1)), (0, 0, 0))
    d = ImageDraw.Draw(out)
    d.text((6, 4), title, fill=(255, 255, 255), font=font(17))
    if subtitle:
        d.text((6, 28), subtitle, fill=(210, 210, 210), font=font(14))
    y = head
    small = font(13)
    for row, h in zip(rows, heights):
        x = 0
        for im, caption in row:
            d.text((x + 4, y + 3), caption, fill=(255, 255, 255), font=small)
            out.paste(im, (x, y + cap))
            x += im.size[0] + gap
        y += h + gap
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    out.save(out_png + '.part.png')
    os.replace(out_png + '.part.png', out_png)
    return out_png


def review_signature(paths, extra=None):
    """Signature of the inputs of a review image set (file stamps + module code)."""
    from src.core.provenance import file_sha256, file_stamp
    return {'inputs': {str(p): file_stamp(p) for p in paths if p and os.path.isfile(p)},
            'code': file_sha256(os.path.abspath(__file__)),
            'extra': extra or {}}


def images_current(manifest, signature):
    """True when the PNGs listed in the manifest exist and were drawn from the same inputs."""
    if not os.path.isfile(manifest):
        return False
    with open(manifest, encoding='utf-8') as handle:
        saved = json.load(handle)
    return (saved.get('signature') == json.loads(json.dumps(signature))
            and all(os.path.isfile(p) for p in saved.get('images', [])))


def save_images_manifest(manifest, signature, images):
    from .common import save_json
    save_json(manifest, {'signature': signature, 'images': list(images)})


def clear_pngs(folder):
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            if name.endswith('.png'):
                os.remove(os.path.join(folder, name))

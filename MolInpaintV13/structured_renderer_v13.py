"""Placement scorer of the V13 text stage.

A supplied label string is rendered in a declared font (DejaVu Sans, sizes 11 and 12, at 64
sub-pixel phases) and every placement inside the search window is described by ten features
of visible fit and position. A small MLP scores the placements; the best one that passes the
visible support and error checks is painted into the covered pixels of its glyph footprint.
Otherwise the region is left to the page model. Nothing here recognises text or reads a
clean image.
"""
from functools import lru_cache
from pathlib import Path
import hashlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
import torch.nn.functional as F


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GlyphSource:
    def __init__(self, font, sizes=(11, 12), phases=64):
        from fontTools.ttLib import TTFont
        self.path = str(Path(font).resolve())
        self.sizes = tuple(int(s) for s in sizes)
        self.phases = int(phases)
        if self.phases < 1:
            raise ValueError('Supply a positive number of raster phases')
        if not self.sizes or min(self.sizes) < 1:
            raise ValueError('Supply positive font sizes')
        with TTFont(self.path) as face:
            self.characters = set(face.getBestCmap())

    @lru_cache(maxsize=2048)
    def raster(self, text, size, phase=0.):
        if not text.strip() or any(c in text for c in '\n\r\t'):
            raise ValueError('Supply one nonempty text line')
        missing = sorted({c for c in text if ord(c) not in self.characters})
        if missing:
            raise ValueError(f'The declared font does not support {missing!r}')
        font = ImageFont.truetype(self.path, int(size))
        x0, y0, x1, y1 = font.getbbox(text)
        canvas = Image.new('L', (x1 - x0 + 8, y1 - y0 + 8))
        ImageDraw.Draw(canvas).text((4 - x0 + phase, 4 - y0), text, font=font, fill=255)
        return np.asarray(canvas.crop(canvas.getbbox()), np.float32).copy() / 255.

    @lru_cache(maxsize=128)
    def variants(self, text):
        # Fractional text origins change FreeType's raster even before any page
        # resizing. Keep every distinct raster on its 1/64-pixel positioning grid.
        out = []
        for size in self.sizes:
            seen = set()
            for k in range(self.phases):
                phase = k / self.phases
                raster = self.raster(text, size, phase)
                key = (raster.shape, raster.tobytes())
                if key not in seen:
                    seen.add(key)
                    out.append((size, phase, raster))
        return out


class PoseScorer(nn.Module):
    """Shared, phrase-independent scorer of visible fit and placement evidence."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 48), nn.SiLU(), nn.Linear(48, 48),
                                 nn.SiLU(), nn.Linear(48, 1))

    def forward(self, features):
        return self.net(features).squeeze(-1)


def candidates(observed, weights, glyphs, text, box=None):
    """Enumerate positions of this string only. Hidden observations are ignored.

    `observed` is the SUM of known ink at each native pixel, `weights` the
    corresponding observation counts. They also support exact nearest-neighbour
    page sampling without resizing damaged pixels back into invented evidence.
    """
    height, width = weights.shape
    if observed.shape != weights.shape or min(height, width) < 1:
        raise ValueError('Invalid observation geometry')
    observed = np.where(weights > 0, observed, 0).astype(np.float32)
    weights = np.asarray(weights, np.float32)
    energy = float((observed ** 2 / np.maximum(weights, 1)).sum())
    if energy < 1e-6:
        return None
    x0, y0, x1, y1 = box if box is not None else (0, 0, width, height)
    bh, bw = max(1, y1 - y0), max(1, x1 - x0)
    features, groups = [], []
    offset = 0
    native_energy = observed ** 2 / np.maximum(weights, 1)
    for size, phase, raster in glyphs.variants(text):
        h, w = raster.shape
        if h > height or w > width:
            continue
        cross = cv2.matchTemplate(observed, raster, cv2.TM_CCORR)
        expected = cv2.matchTemplate(weights, raster ** 2, cv2.TM_CCORR)
        # The search margin can contain node borders and arrows. Score the full
        # candidate rectangle, including its internal whitespace, without charging
        # unrelated strokes elsewhere in that margin to the text hypothesis.
        local_energy = cv2.matchTemplate(native_energy, np.ones_like(raster), cv2.TM_CCORR)
        denominator = np.maximum(local_energy, 1.)
        error = np.maximum(0, (expected - 2 * cross + local_energy) / denominator)
        support = cv2.matchTemplate((weights > 0).astype(np.float32), raster,
                                   cv2.TM_CCORR) / max(float(raster.sum()), 1)
        yy, xx = np.indices(error.shape, dtype=np.float32)
        fx = (xx + w / 2 - (x0 + x1) / 2) / bw
        fy = (yy + h / 2 - (y0 + y1) / 2) / bh
        row = np.stack((error.clip(0, 8), (cross / denominator).clip(0, 8),
                        (expected / denominator).clip(0, 8), support.clip(0, 1),
                        fx, fy, fx ** 2, fy ** 2,
                        np.full_like(fx, np.log(w / bw)),
                        np.full_like(fx, np.log(h / bh))), -1).reshape(-1, 10)
        features.append(row)
        groups.append(dict(start=offset, stop=offset + len(row), shape=error.shape,
                           raster=raster, font_size=size, phase=phase))
        offset += len(row)
    if not groups:
        return None
    return dict(features=np.concatenate(features).astype(np.float32), groups=groups,
                shape=(height, width))


def hard_raster(pack, index):
    for group in pack['groups']:
        if group['start'] <= index < group['stop']:
            y, x = np.unravel_index(index - group['start'], group['shape'])
            raster = group['raster']
            out = np.zeros(pack['shape'], np.float32)
            out[y:y + raster.shape[0], x:x + raster.shape[1]] = raster
            return out, dict(x=int(x), y=int(y), font_size=group['font_size'], phase=group['phase'])
    raise IndexError(index)


class StructuredRenderer:
    def __init__(self, scorer, glyphs, minimum_support=.15, maximum_error=.25):
        self.scorer, self.glyphs = scorer, glyphs
        self.minimum_support, self.maximum_error = minimum_support, maximum_error

    @classmethod
    def load(cls, checkpoint, font, device='cpu'):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved.get('format') != 'molinpaint_v13_placement_scorer':
            raise ValueError('Expected a V13 placement scorer checkpoint')
        if digest(font) != saved['font_sha256']:
            raise ValueError('The glyph font differs from the one the scorer was trained with')
        scorer = PoseScorer().to(device)
        scorer.load_state_dict(saved['model'], strict=True)
        scorer.eval()
        return cls(scorer, GlyphSource(font, saved['font_sizes'], saved['phases']))

    @torch.no_grad()
    def choose(self, pack):
        device = next(self.scorer.parameters()).device
        features = torch.as_tensor(pack['features'], device=device)
        scores = self.scorer(features)
        # These are observable support checks, not scores against a clean image.
        valid = (features[:, 3] >= self.minimum_support) & (features[:, 0] <= self.maximum_error)
        if not valid.any():
            return None, dict(reason='insufficient_visible_fit')
        index = int(scores.masked_fill(~valid, -torch.inf).argmax())
        raster, info = hard_raster(pack, index)
        return raster, dict(**info, reason='accepted',
                            visible_error=float(features[index, 0]),
                            visible_support=float(features[index, 3]))

    def repair(self, occ, mask, base, box, text, geometry=(1., 1., 0., 0.)):
        """Repair one supplied line; all arrays are 2D and ink=1, paper=0.

        Geometry is (scale_x, scale_y, offset_x, offset_y) of the nearest-resized
        native page. No target image, transcript annotation, or phrase list is read.
        """
        if occ.shape != mask.shape or occ.shape != base.shape:
            raise ValueError('Image, mask and baseline shapes must match')
        result = np.where(mask > .5, base, occ).copy()
        if not text.strip():
            return result, dict(reason='no_supplied_text')
        sx, sy, ox, oy = geometry
        if min(sx, sy) <= 0:
            raise ValueError('Page scales must be positive')
        height, width = occ.shape
        x0, y0, x1, y1 = box
        pad = max(2, round(4 * max(sx, sy)))
        left, top = max(0, int(x0) - pad), max(0, int(y0) - pad)
        right, bottom = min(width, int(np.ceil(x1)) + pad), min(height, int(np.ceil(y1)) + pad)
        if right <= left or bottom <= top:
            raise ValueError('Empty region')
        known = (mask[top:bottom, left:right] <= .5).astype(np.float32)
        visible = occ[top:bottom, left:right] * known
        if float(visible.sum()) < 6 * sx * sy:
            return result, dict(reason='insufficient_visible_ink')
        gx = np.floor((np.arange(left, right) - ox + .5) / sx).astype(int)
        gy = np.floor((np.arange(top, bottom) - oy + .5) / sy).astype(int)
        shape = (int(gy[-1] - gy[0] + 1), int(gx[-1] - gx[0] + 1))
        weights, observations = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        indices = (gy[:, None] - gy[0], gx[None, :] - gx[0])
        np.add.at(weights, indices, known)
        np.add.at(observations, indices, visible)
        local_box = ((x0 - ox) / sx - gx[0], (y0 - oy) / sy - gy[0],
                     (x1 - ox) / sx - gx[0], (y1 - oy) / sy - gy[0])
        pack = candidates(observations, weights, self.glyphs, text, local_box)
        if pack is None:
            return result, dict(reason='no_candidate_geometry')
        raster, info = self.choose(pack)
        if raster is None:
            return result, info
        # Paint the selected glyph footprint plus one native pixel, preserving
        # nearby diagram strokes in the rest of the search window.
        template = self.glyphs.raster(text, info['font_size'], info['phase'])
        xx, yy = gx[None, :] - gx[0], gy[:, None] - gy[0]
        footprint = ((xx >= info['x'] - 1) & (xx <= info['x'] + template.shape[1]) &
                     (yy >= info['y'] - 1) & (yy <= info['y'] + template.shape[0]))
        active = (known == 0) & footprint
        result[top:bottom, left:right] = np.where(active, raster[indices],
                                                 result[top:bottom, left:right])
        return result, dict(**info, text=text, source_x=int(gx[0] + info['x']),
                            source_y=int(gy[0] + info['y']))

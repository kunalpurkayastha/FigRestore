#!/usr/bin/env python3
"""Restore the covered region of a technical drawing (molecule, flowchart or circuit).

Inputs: the damaged image, a mask that is white (255) on covered pixels, and the family.
Output: an image of the same size in which covered pixels are predicted and every other
pixel is copied from the input unchanged.

    python MolInpaintV13/restore_v13.py --image damaged.png --mask mask.png --family flowchart --output restored.png

For one page:
  1. The frozen VLLM reads the page with the covered pixels tinted; the projected hidden
     states of its image tokens are the conditioning.
  2. The coarse U-Net predicts ink at 256 px.
  3. The page adapter of the family refines it at 1024 px and predicts where text is.
  4. Flowcharts and circuits: the VLLM serialises the drawing and the text stage writes its
     label strings into damaged text regions.
  5. The prediction is written into the mask and the letterbox is undone.
"""
import argparse
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(HERE.parent / '.cache' / 'huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from page_model_v13 import load_coarse_model, load_page_model  # noqa: E402
from structured_renderer_v13 import StructuredRenderer  # noqa: E402
from text_stage_v13 import emitted_labels, repair_text  # noqa: E402
from vllm_v13 import PROMPTS, Vllm, tint_covered  # noqa: E402

FAMILIES = tuple(PROMPTS)


def load_config(path=HERE / 'config_v13.yaml'):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg['weights'] = {key: str((path.parent / value).resolve()) for key, value in cfg['weights'].items()}
    return cfg


def letterbox_geometry(size, resolution):
    """Size and offset of an image fitted into a square of side `resolution`, aspect kept."""
    w, h = size
    scale = min(resolution / w, resolution / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    return nw, nh, (resolution - nw) // 2, (resolution - nh) // 2


def letterbox(image, resolution, fill):
    if image.size == (resolution, resolution):
        return image
    nw, nh, dx, dy = letterbox_geometry(image.size, resolution)
    out = Image.new(image.mode, (resolution, resolution), fill)
    out.paste(image.resize((nw, nh), Image.NEAREST), (dx, dy))
    return out


def input_tensors(image, mask, resolution):
    """Signed page (ink -1, paper +1) and binary mask, letterboxed to the page resolution."""
    if image.size != mask.size:
        raise ValueError('Image and mask must have identical original dimensions')
    im = np.asarray(letterbox(image.convert('L'), resolution, 255)).astype(np.float32)
    mk = np.asarray(letterbox(mask.convert('L'), resolution, 0)) > 127
    return torch.from_numpy(im / 127.5 - 1)[None, None], torch.from_numpy(mk.astype(np.float32))[None, None]


def restore_canvas(prediction, image, mask):
    """Undo the letterbox, then copy the original RGB pixels outside the mask byte for byte."""
    res = prediction.shape[-1]
    nw, nh, dx, dy = letterbox_geometry(image.size, res)
    gray = ((prediction.numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    crop = Image.fromarray(gray[dy:dy + nh, dx:dx + nw]).resize(image.size, Image.Resampling.BILINEAR)
    repaired = np.repeat(np.asarray(crop)[:, :, None], 3, axis=2)
    original = np.asarray(image.convert('RGB'))
    return Image.fromarray(np.where((np.asarray(mask.convert('L')) > 127)[:, :, None], repaired, original))


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


class Restorer:
    """Loads every model once; call restore() for each page."""

    def __init__(self, device='cuda', config=HERE / 'config_v13.yaml'):
        self.device = torch.device(device)
        self.cfg = load_config(config)
        weights = self.cfg['weights']
        self.adapters = {name: load_page_model(weights[name], self.device)
                         for name in sorted(set(self.cfg['page_adapter'].values()))}
        self.renderer = StructuredRenderer.load(weights['placement_scorer'], weights['font'])
        self.vllm = Vllm(weights['vllm'], self.cfg['vllm'], self.device)
        self.coarse = load_coarse_model(weights['coarse_model'], self.device)
        if self.coarse.config['hidden_size'] != self.vllm.hidden_size:
            raise ValueError('The coarse model was trained on a VLLM with a different hidden size')
        if any(a.config['cross_attention_dim'] != self.coarse.config['cross_attention_dim'] for a in self.adapters.values()):
            raise ValueError('Page adapters and coarse model disagree on the conditioning width')
        resolutions = {a.resolution for a in self.adapters.values()}
        if len(resolutions) != 1:
            raise ValueError('Page adapters were trained at different resolutions')
        self.resolution = resolutions.pop()
        logging.getLogger('transformers').setLevel(logging.ERROR)

    def restore(self, image, mask, family):
        if family not in FAMILIES:
            raise ValueError(f'family must be one of {FAMILIES}')
        image, mask = image.convert('RGB'), mask.convert('L')
        if image.size != mask.size:
            raise ValueError('Image and mask must have the same size')
        device = self.device
        damaged = image.convert('L')
        tinted = tint_covered(damaged, mask, self.cfg['vllm']['mask_overlay'])
        occluded, covered = input_tensors(damaged, mask, self.resolution)
        inputs = self.vllm.inputs(tinted, family)
        with torch.no_grad(), autocast(device):
            hidden = self.vllm.hidden_states(inputs)
            cross, enc_mask = self.coarse.condition(hidden, inputs['attention_mask'], inputs['images_seq_mask'])
            del hidden
            coarse = self.coarse(cross, enc_mask, occluded.to(device), covered.to(device))
        # The page adapters were trained on conditioning and coarse logits stored in float16.
        coarse = coarse[0].half().float().cpu()[None].to(device)
        cross = cross[0, enc_mask[0]].half().float().cpu()[None].to(device)
        enc_mask = torch.ones(cross.shape[:2], dtype=torch.bool, device=device)
        text_stage = family in self.cfg['text_stage']
        serialisation = self.vllm.serialise(tinted, family) if text_stage else ''
        labels = emitted_labels(family, serialisation) if serialisation else []
        page = self.adapters[self.cfg['page_adapter'][family]]
        occluded, covered = occluded.to(device), covered.to(device)
        with torch.no_grad():
            base, location = page(occluded, covered, coarse, cross, enc_mask)
            if text_stage:
                ink, report = repair_text(self.renderer, base, location, occluded, covered, self.cfg['detector'],
                                          labels, source_size=image.size, ocr=page.ocr)
            else:
                ink, report = base, {'boxes': [], 'repairs': []}
        known = (1 - occluded) * .5
        outside = ~covered.bool()
        if not torch.equal(ink[outside], known[outside]):
            raise RuntimeError('Known pixels changed; refusing to write output')
        restored = restore_canvas((1 - 2 * ink)[0, 0].float().cpu(), image, mask)
        repairs = report['repairs']
        return dict(image=restored, ink=ink, page=base, serialisation=serialisation, labels=labels,
                    regions=len(report['boxes']), repairs=repairs,
                    accepted=sum(1 for r in repairs if r.get('reason') == 'accepted'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', required=True, help='Damaged image')
    ap.add_argument('--mask', required=True, help='Mask, white (255) on covered pixels, same size as the image')
    ap.add_argument('--family', required=True, choices=FAMILIES)
    ap.add_argument('--output', required=True, help='Restored image')
    ap.add_argument('--details', help='Optional JSON with the serialisation, labels and text-stage decisions')
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--config', default=str(HERE / 'config_v13.yaml'))
    args = ap.parse_args()
    image, mask = Image.open(args.image).convert('RGB'), Image.open(args.mask).convert('L')
    if image.size != mask.size:
        raise SystemExit('Image and mask must have the same size')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not (np.asarray(mask) > 127).any():
        image.save(output)
        print(f'No covered pixels in the mask; copied the input to {output}')
        return
    result = Restorer(args.device, args.config).restore(image, mask, args.family)
    result['image'].save(output)
    if args.details:
        details = {k: result[k] for k in ('serialisation', 'labels', 'regions', 'accepted', 'repairs')}
        Path(args.details).write_text(json.dumps(details, indent=1,
                                                 default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print(f'Wrote {output} ({image.size[0]}x{image.size[1]}); text regions {result["regions"]}, '
          f'accepted repairs {result["accepted"]}')


if __name__ == '__main__':
    main()

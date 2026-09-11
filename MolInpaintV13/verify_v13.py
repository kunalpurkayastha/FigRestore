#!/usr/bin/env python3
"""End-to-end check of the V13 system on the bundled validation examples.

Each example holds a damaged page, its mask and the clean page, plus the masked F1 and the
number of accepted text repairs recorded when the full validation split was evaluated. The
check passes when every page reproduces those values and no visible pixel changes.

    python MolInpaintV13/verify_v13.py
"""
import argparse
import json
import sys
import time

import numpy as np
from PIL import Image
import torch

from restore_v13 import HERE, Restorer, input_tensors

ROOT = HERE.parent


def masked_f1(pred, gt, region):
    tp = float((pred & gt & region).sum())
    fp = float((pred & ~gt & region).sum())
    fn = float((~pred & gt & region).sum())
    return None if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--tolerance', type=float, default=1e-3)
    args = ap.parse_args()
    began = time.monotonic()
    restorer = Restorer(args.device)
    load_seconds = time.monotonic() - began
    rows, ok = [], True
    for folder in sorted(p for p in (ROOT / 'examples').iterdir() if (p / 'info.json').is_file()):
        info = json.loads((folder / 'info.json').read_text())
        image = Image.open(folder / 'damaged.png').convert('RGB')
        mask = Image.open(folder / 'mask.png').convert('L')
        clean = Image.open(folder / 'clean.png').convert('L')
        start = time.monotonic()
        result = restorer.restore(image, mask, info['family'])
        seconds = time.monotonic() - start
        result['image'].save(folder / 'restored.png')
        gt_signed, covered = input_tensors(clean, mask, restorer.resolution)
        gt = ((1 - gt_signed[0, 0]) * .5).numpy() > .5
        region = covered[0, 0].numpy() > .5
        score = masked_f1(result['ink'][0, 0].float().cpu().numpy() > .5, gt, region)
        visible = np.asarray(mask) <= 127
        known_ok = np.array_equal(np.asarray(result['image'])[visible], np.asarray(image)[visible])
        expected = info['expected_masked_f1_final']
        passed = (score is not None and abs(score - expected) <= args.tolerance and known_ok
                  and result['accepted'] == info['expected_accepted_repairs'])
        ok &= passed
        rows.append(dict(example=folder.name, family=info['family'], masked_f1=score, expected=expected,
                         accepted=result['accepted'], expected_accepted=info['expected_accepted_repairs'],
                         known_pixels_unchanged=known_ok, seconds=round(seconds, 1), passed=passed))
        print(f"{'PASS' if passed else 'FAIL'}  {folder.name:46s} F1 {score:.4f} (expected {expected:.4f})  "
              f"accepted {result['accepted']}/{info['expected_accepted_repairs']}  known pixels "
              f"{'unchanged' if known_ok else 'CHANGED'}  {seconds:.1f}s", flush=True)
    report = dict(passed=bool(ok), device=args.device, load_seconds=round(load_seconds, 1), examples=rows)
    if args.device == 'cuda':
        report.update(gpu=torch.cuda.get_device_name(0),
                      peak_gpu_memory_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                      torch_arch_list=torch.cuda.get_arch_list())
    (ROOT / 'verification.json').write_text(json.dumps(report, indent=1))
    print('ALL PASSED' if ok else 'SOME CHECKS FAILED', '| report: verification.json')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

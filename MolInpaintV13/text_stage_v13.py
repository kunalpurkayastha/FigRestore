"""Text stage of the V13 system (flowcharts and circuits).

Label strings from the VLLM serialisation are written into damaged text regions:

    serialisation      -> (grid cell, label) pairs, sorted in grid order
    text-location map  -> text regions (connected components), sorted in reading order
    undamaged regions  -> CRNN reads; a read that matches a label anchors the alignment
    monotone alignment -> one label, or none, for every damaged region
    placement scorer   -> fits the label to the visible pixels of its region, or abstains

Assignment is by order, not by nearest position. The VLLM emits a grid cell per label and
the regions are read top to bottom, so a Needleman-Wunsch alignment tolerates a missed or an
extra region without shifting every later label. No annotation, transcript or clean pixel is
read here.
"""
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F

SKIP = 1.0                          # cost of leaving a region unlabelled or a label unplaced
READ_HEIGHT, READ_WIDTH = 32, 320   # CRNN crop size


# ------------------------------------------------------------------------ serialisation
def _parse_flowchart(text):
    """`FLOW|r0c0:kind:label;...|r0c0>r1c0:condition,...` -> nodes and edges, or None."""
    try:
        tag, nodes_s, edges_s = text.split("|", 2)
        if tag != "FLOW":
            return None
        nodes = {}
        for tok in filter(None, nodes_s.split(";")):
            parts = tok.split(":", 2)
            if len(parts) == 3:
                nodes[parts[0]] = {"kind": parts[1], "label": parts[2]}
            elif len(parts) == 2:
                nodes[parts[0]] = {"kind": "process", "label": parts[1]}
        edges = []
        for t in filter(None, edges_s.split(",")):
            body, _, lab = t.partition(":")
            if ">" in body:
                a, b = body.split(">", 1)
                edges.append((a, b, lab))
        return {"nodes": nodes, "edges": edges}
    except Exception:
        return None


def _parse_circuit(text):
    """`CIRC|r0c0:kind:orient[:value];...|pin+pin,...` -> components and nets, or None."""
    try:
        tag, comp_s, net_s = text.split("|", 2)
        if tag != "CIRC":
            return None
        comps = {}
        for tok in filter(None, comp_s.split(";")):
            p = tok.split(":")
            if len(p) >= 3:
                comps[p[0]] = {"kind": p[1], "orient": p[2], "value": p[3] if len(p) > 3 else ""}
        nets = [tuple(sorted(t.split("+"))) for t in filter(None, net_s.split(",")) if "+" in t]
        return {"components": comps, "nets": nets}
    except Exception:
        return None


def emitted_labels(family, serialisation):
    """[(grid cell, label)] carried by a serialisation; empty if it does not parse.

    Flowcharts: node labels and edge conditions. Circuits: component values. Molecules
    (SMILES) carry no rendered label text.
    """
    text = (serialisation or '').strip()
    if family == 'flowchart':
        parsed = _parse_flowchart(text)
    elif family == 'circuit':
        parsed = _parse_circuit(text)
    else:
        return []
    if parsed is None:
        return []
    if family == 'flowchart':
        out = [(key, node.get('label', '').strip()) for key, node in parsed['nodes'].items()]
        out += [(f'{a}>{b}', label.strip()) for a, b, label in parsed['edges'] if label]
    else:
        out = [(key, component.get('value', '').strip()) for key, component in parsed['components'].items()]
    return [(key, value) for key, value in out if value]


# ------------------------------------------------------------------------ regions
def text_regions(location, detector):
    """Boxes [x0, y0, x1, y1] of the connected components of the predicted text-location map."""
    probability = location.detach().float().sigmoid().cpu().numpy().squeeze()
    binary = probability > detector['threshold']
    binary = ndimage.binary_closing(binary, structure=np.ones((3, 3), dtype=bool))
    labels, _ = ndimage.label(binary)
    height, width = binary.shape
    out = []
    for slices in ndimage.find_objects(labels):
        if slices is None:
            continue
        ys, xs = slices
        if (ys.stop - ys.start) * (xs.stop - xs.start) < detector['min_area']:
            continue
        if ys.stop - ys.start > height * detector['max_height_fraction']:
            continue
        pad = detector['padding']
        out.append([max(0, xs.start - pad), max(0, ys.start - pad),
                    min(width, xs.stop + pad), min(height, ys.stop + pad)])
    return out


def crop_rois(images, boxes, height=READ_HEIGHT, width=READ_WIDTH):
    """Bilinear crops of half-open pixel boxes given as [batch, x0, y0, x1, y1]."""
    if not len(boxes):
        return images.new_empty((0, images.shape[1], height, width))
    boxes = boxes.to(device=images.device, dtype=torch.float32)
    _, _, h, w = images.shape
    ys = (torch.arange(height, device=images.device, dtype=torch.float32) + .5) / height
    xs = (torch.arange(width, device=images.device, dtype=torch.float32) + .5) / width
    x = boxes[:, 1, None] + (boxes[:, 3] - boxes[:, 1])[:, None] * xs
    y = boxes[:, 2, None] + (boxes[:, 4] - boxes[:, 2])[:, None] * ys
    grid = torch.stack((x[:, None, :].expand(-1, height, -1) * 2 / w - 1,
                        y[:, :, None].expand(-1, -1, width) * 2 / h - 1), dim=-1)
    return F.grid_sample(images[boxes[:, 0].long()].float(), grid,
                         mode="bilinear", padding_mode="border", align_corners=False)


def read_regions(ink, boxes, mask, ocr, max_damage=.02):
    """CRNN read of every region that is essentially undamaged; None elsewhere."""
    covered = mask[0, 0].detach().float().cpu().numpy()
    out = [None] * len(boxes)
    wanted = [i for i, (x0, y0, x1, y1) in enumerate(boxes)
              if covered[y0:y1, x0:x1].size and covered[y0:y1, x0:x1].mean() <= max_damage]
    if not wanted:
        return out
    rois = torch.tensor([[0., *boxes[i]] for i in wanted], dtype=torch.float32, device=ink.device)
    with torch.no_grad():
        logits, _ = ocr(crop_rois(ink, rois, READ_HEIGHT, READ_WIDTH))
    for index, text in zip(wanted, ocr.decode(logits)):
        out[index] = text
    return out


# ------------------------------------------------------------------------ alignment
def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        nxt = [i]
        for j, cb in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = nxt
    return row[-1]


def text_distance(a, b):
    """Normalised edit distance, tolerant of the line wrapping in node labels."""
    a, b = ' '.join(a.split()), ' '.join(b.split())
    if not a or not b:
        return 1.
    if a in b or b in a:
        return .05
    return min(1., edit_distance(a, b) / max(len(a), len(b)))


def reading_order(boxes, line_tolerance=.6):
    """Sort boxes top to bottom, then left to right, tolerating baseline jitter."""
    if not boxes:
        return []
    heights = [b[3] - b[1] for b in boxes]
    tolerance = max(1., float(np.median(heights)) * line_tolerance)
    return sorted(range(len(boxes)), key=lambda i: (round(boxes[i][1] / tolerance), boxes[i][0]))


def grid_key(cell):
    """`r3c1` -> (3, 1); anything unparsable sorts last but keeps its order."""
    try:
        row, column = cell.lstrip('r').split('c')
        return int(row), int(column)
    except (ValueError, AttributeError):
        return 10 ** 6, 10 ** 6


def wrap_cost(box, text, characters_per_pixel=.09):
    """Lenient cost that discourages a long label in a box that could not hold any of it."""
    width = max(1., box[2] - box[0])
    expected = max(1., len(text) * characters_per_pixel * (box[3] - box[1]))
    return min(1., abs(np.log(expected / width)) * .25)


def align(boxes, labels, cost_fn=None, observations=None, anchor_weight=3.0):
    """Monotone alignment of ordered boxes to ordered labels.

    `observations` holds one CRNN read per box (None where the box is damaged). A read that
    matches a label costs almost nothing to pair, a mismatch is expensive, so the undamaged
    text holds the alignment and the damaged regions fall into the slots between.
    Returns a list the length of `boxes`, each entry a label index or None.
    """
    order = reading_order(boxes)
    ordered_labels = sorted(range(len(labels)), key=lambda i: grid_key(labels[i][0]))
    n, m = len(order), len(ordered_labels)
    if not n or not m:
        return [None] * len(boxes)
    table = np.zeros((n + 1, m + 1), dtype=np.float64)
    table[:, 0] = np.arange(n + 1) * SKIP
    table[0, :] = np.arange(m + 1) * SKIP
    pairs = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        box = boxes[order[i]]
        read = None if observations is None else observations[order[i]]
        for j in range(m):
            text = labels[ordered_labels[j]][1]
            value = 0. if cost_fn is None else cost_fn(box, text)
            if read:
                value += anchor_weight * text_distance(read, text)
            pairs[i, j] = value
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            table[i, j] = min(table[i - 1, j - 1] + pairs[i - 1, j - 1],
                              table[i - 1, j] + SKIP, table[i, j - 1] + SKIP)
    out = [None] * len(boxes)
    i, j = n, m
    while i > 0 and j > 0:
        pair = pairs[i - 1, j - 1]
        if abs(table[i, j] - (table[i - 1, j - 1] + pair)) < 1e-9:
            out[order[i - 1]] = ordered_labels[j - 1]
            i, j = i - 1, j - 1
        elif abs(table[i, j] - (table[i - 1, j] + SKIP)) < 1e-9:
            i -= 1
        else:
            j -= 1
    return out


@torch.no_grad()
def assign_labels(location, occluded, mask, detector, labels, ocr=None):
    """Choose a label (or none) for every damaged text region."""
    # Align over every region, damaged or not: the undamaged ones are the anchors.
    everything = text_regions(location, detector)
    covered = mask[0, 0].detach().float().cpu().numpy()
    is_damaged = [covered[y0:y1, x0:x1].size > 0 and covered[y0:y1, x0:x1].max() > .5
                  for x0, y0, x1, y1 in everything]
    boxes = [box for box, flag in zip(everything, is_damaged) if flag][:detector['max_regions']]
    report = {'boxes': boxes, 'assigned': [], 'strings': [], 'anchors': 0, 'regions': len(everything)}
    if not boxes:
        return report
    anchors = read_regions((1 - occluded) * .5, everything, mask, ocr) if ocr is not None else None
    report['anchors'] = sum(1 for a in (anchors or []) if a)
    mapping = align(everything, labels, cost_fn=wrap_cost, observations=anchors)
    keep = [i for i, flag in enumerate(is_damaged) if flag][:detector['max_regions']]
    report['assigned'] = [None if mapping[i] is None else labels[mapping[i]][0] for i in keep]
    report['strings'] = [labels[mapping[i]][1] if mapping[i] is not None else '' for i in keep]
    return report


@torch.no_grad()
def repair_text(renderer, base, location, occluded, mask, detector, labels, source_size, ocr=None):
    """Write assigned labels into the damaged regions of one page.

    `base` is the page model's ink (1024 px, ink 1, paper 0); `source_size` is the original
    image's (width, height) before letterboxing. Returns the ink and the decisions made.
    """
    if len(base) != 1:
        raise ValueError('The text stage expects one page at a time')
    report = assign_labels(location, occluded, mask, detector, labels, ocr=ocr)
    sw, sh = source_size
    height, width = base.shape[-2:]
    factor = min(width / sw, height / sh)
    nw, nh = max(1, round(sw * factor)), max(1, round(sh * factor))
    geometry = (nw / sw, nh / sh, (width - nw) // 2, (height - nh) // 2)
    known = (1 - occluded) * .5
    occ = known[0, 0].float().cpu().numpy()
    damaged = mask[0, 0].float().cpu().numpy()
    result = base[0, 0].float().cpu().numpy().copy()
    repairs = []
    for box, text in zip(report['boxes'], report['strings']):
        result, info = renderer.repair(occ, damaged, result, box, text, geometry)
        repairs.append(info)
    ink = torch.from_numpy(result).to(device=base.device, dtype=base.dtype)[None, None]
    report['repairs'] = repairs
    return torch.where(mask.bool(), ink, known), report

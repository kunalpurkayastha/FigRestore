"""Page reconstruction of the V13 system.

    VLLM image-token hidden states -> projection (LayerNorm, Linear 1280->768, LayerNorm)
    damaged page at 256 px + conditioning -> coarse U-Net -> coarse ink logits
    damaged page at 1024 px + upsampled coarse logits + conditioning + CRNN features
        -> page adapter -> ink probability inside the mask, text-location logits

Two page adapters are shipped. The base adapter is used for molecules and flowcharts. The
refined adapter starts from it, adds attention from its 256 px decoder features to the
conditioning and was trained further with text losses; it is used for circuits. Both add the
features of the frozen CRNN text recogniser, gated by the predicted text location.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

COARSE_FORMAT = 'molinpaint_v13_coarse_model'
ADAPTER_FORMAT = 'molinpaint_v13_page_adapter'


def load_checkpoint(path, expected_format):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format') != expected_format:
        raise ValueError(f'{path} is not a {expected_format} checkpoint')
    return saved


# ------------------------------------------------------------------------ coarse model
class CoarseModel(nn.Module):
    """Projection of the VLLM hidden states and the coarse U-Net (regression head)."""

    def __init__(self, config):
        from diffusers import UNet2DConditionModel
        super().__init__()
        self.config = dict(config)
        hidden, cross = int(config['hidden_size']), int(config['cross_attention_dim'])
        channels = tuple(int(c) for c in config['block_out_channels'])
        levels, attention = len(channels), int(config['attn_levels'])
        down = tuple('CrossAttnDownBlock2D' if i >= levels - attention else 'DownBlock2D' for i in range(levels))
        up = tuple('CrossAttnUpBlock2D' if i < attention else 'UpBlock2D' for i in range(levels))
        self.unet = UNet2DConditionModel(
            sample_size=int(config['resolution']), in_channels=2, out_channels=1,
            cross_attention_dim=cross, block_out_channels=channels,
            down_block_types=down, up_block_types=up,
            layers_per_block=int(config['layers_per_block']),
            norm_num_groups=int(config['norm_num_groups']),
            attention_head_dim=int(config['attention_head_dim']))
        self.proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, cross), nn.LayerNorm(cross))

    @torch.no_grad()
    def condition(self, hidden_states, attention_mask, images_seq_mask):
        """Projected hidden states of the image tokens, packed to the left, and their mask."""
        layer = int(self.config['hidden_layer'])
        if not -len(hidden_states) <= layer < len(hidden_states):
            raise ValueError(f'hidden_layer={layer} is outside the VLLM hidden-state list')
        cross = self.proj(hidden_states[layer].to(self.proj[0].weight.dtype))
        selected = attention_mask.bool() & images_seq_mask.bool()
        if not selected.any(1).all():
            raise ValueError('A page has no image tokens')
        rows = [x[m] for x, m in zip(cross, selected)]
        packed = pad_sequence(rows, batch_first=True)
        lengths = torch.tensor([len(x) for x in rows], device=cross.device)
        return packed, torch.arange(packed.shape[1], device=cross.device)[None] < lengths[:, None]

    @torch.no_grad()
    def forward(self, cross, enc_mask, occluded, mask):
        """Coarse ink logits. The page is erased under the mask before it is downsampled."""
        resolution = int(self.config['resolution'])
        coarse_mask = F.adaptive_max_pool2d(mask.float(), (resolution, resolution))
        coarse_known = F.interpolate(occluded.float() * (1 - mask.float()), (resolution, resolution), mode='area')
        coarse_known = coarse_known * (1 - coarse_mask)
        timestep = torch.zeros(len(occluded), device=occluded.device, dtype=torch.long)
        return self.unet(torch.cat((coarse_known, coarse_mask), 1), timestep, encoder_hidden_states=cross,
                         encoder_attention_mask=enc_mask).sample


def load_coarse_model(path, device):
    saved = load_checkpoint(path, COARSE_FORMAT)
    model = CoarseModel(saved['config'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).requires_grad_(False).eval()


# ------------------------------------------------------------------------ page adapter
def conv_block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.GroupNorm(4, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU())


def _up(value, skip):
    return torch.cat((F.interpolate(value, skip.shape[-2:], mode='bilinear', align_corners=False), skip), 1)


class FineContext(nn.Module):
    """Attention from 256 px decoder features to the conditioning (refined adapter only)."""

    def __init__(self, dim, cross_dim, chunk=2048):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.context = nn.Linear(cross_dim, dim)
        self.attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.chunk = int(chunk)

    def forward(self, feature, cross, enc_mask):
        q = feature.flatten(2).transpose(1, 2)
        kv = self.context(cross)
        pieces = [self.attention(self.norm(part), kv, kv, key_padding_mask=~enc_mask.bool(),
                                 need_weights=False)[0] for part in q.split(self.chunk, dim=1)]
        return feature + torch.cat(pieces, 1).transpose(1, 2).reshape_as(feature)


class PageAdapter(nn.Module):
    """U-Net at 1024 px that corrects the upsampled coarse logits and locates text."""

    def __init__(self, cross_dim, width=16, heads=4, fine_attention=False, query_chunk=2048):
        super().__init__()
        self.enc0 = conv_block(5, width)
        self.enc1 = conv_block(width, width * 2, 2)
        self.enc2 = conv_block(width * 2, width * 4, 2)
        self.enc3 = conv_block(width * 4, width * 8, 2)
        dim = width * 8
        self.ocr_project = nn.Conv2d(128, dim, 1)
        self.location = nn.Conv2d(dim, 1, 1)
        self.query_norm = nn.LayerNorm(dim)
        self.context = nn.Linear(cross_dim, dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.dec2 = conv_block(dim + width * 4, width * 4)
        self.dec1 = conv_block(width * 4 + width * 2, width * 2)
        self.dec0 = conv_block(width * 2 + width, width)
        self.output = nn.Conv2d(width, 3, 1)  # geometry correction, text correction, fine location
        self.fine_context = FineContext(width * 4, cross_dim, query_chunk) if fine_attention else None

    def forward(self, known, mask, coarse, cross, enc_mask, ocr_features):
        batch, _, height, width = known.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=known.device, dtype=known.dtype),
                                torch.linspace(-1, 1, width, device=known.device, dtype=known.dtype), indexing='ij')
        coords = torch.stack((xx, yy))[None].expand(batch, -1, -1, -1)
        a = self.enc0(torch.cat((known, mask, coarse.sigmoid(), coords), 1))
        b = self.enc1(a)
        c = self.enc2(b)
        d = self.enc3(c)
        q = d.flatten(2).transpose(1, 2)
        kv = self.context(cross)
        update = self.attention(self.query_norm(q), kv, kv, key_padding_mask=~enc_mask.bool(), need_weights=False)[0]
        d = d + update.transpose(1, 2).reshape_as(d)
        location = self.location(d)
        if ocr_features is not None:
            # CRNN features enter where the model expects text.
            local = F.interpolate(ocr_features, d.shape[-2:], mode='bilinear', align_corners=False)
            d = d + location.sigmoid() * self.ocr_project(local)
        d = self.dec2(_up(d, c))
        if self.fine_context is not None:
            d = self.fine_context(d, cross, enc_mask)
        d = self.dec1(_up(d, b))
        d = self.dec0(_up(d, a))
        geometry, text, fine_location = self.output(d).chunk(3, 1)
        location = F.interpolate(location, (height, width), mode='bilinear', align_corners=False) + fine_location
        return coarse + geometry + location.sigmoid() * text, location


class TextRecognizer(nn.Module):
    """Small CRNN trained on drawing text; blank is class 0. Input: ink probability (paper 0, ink 1)."""

    def __init__(self, alphabet):
        super().__init__()
        self.alphabet = alphabet
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=(2, 1), padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.SiLU())
        self.sequence = nn.GRU(128, 128, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256, len(alphabet) + 1)

    def forward(self, ink):
        feature = self.encoder(ink)
        sequence, _ = self.sequence(feature.mean(2).transpose(1, 2))
        return self.classifier(sequence).transpose(0, 1), feature

    @torch.no_grad()
    def decode(self, logits):
        """Greedy CTC decoding of [time, batch, classes] logits."""
        result = []
        for row in logits.argmax(-1).T.tolist():
            last, chars = -1, []
            for token in row:
                if token and token != last:
                    chars.append(self.alphabet[token - 1])
                last = token
            result.append(''.join(chars))
        return result


class PageModel(nn.Module):
    """One page adapter with its CRNN."""

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.text_adapter = PageAdapter(int(config['cross_attention_dim']), int(config['width']),
                                        int(config['attention_heads']), bool(config['fine_attention']),
                                        int(config['query_chunk']))
        self.ocr = TextRecognizer(config['alphabet'])

    @property
    def resolution(self):
        return int(self.config['resolution'])

    @torch.no_grad()
    def forward(self, occluded, mask, coarse, cross, enc_mask):
        """Ink probability (known pixels copied from the page) and text-location logits."""
        known = (1 - occluded.float()) * .5 * (1 - mask.float())
        features = self.ocr.encoder(known) if self.config.get('use_ocr_features', True) else None
        up = F.interpolate(coarse.float(), occluded.shape[-2:], mode='bilinear', align_corners=False)
        logits, location = self.text_adapter(known, mask, up, cross, enc_mask, features)
        return torch.where(mask.bool(), logits.sigmoid(), (1 - occluded) * .5), location


def load_page_model(path, device):
    saved = load_checkpoint(path, ADAPTER_FORMAT)
    model = PageModel(saved['config'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).requires_grad_(False).eval()

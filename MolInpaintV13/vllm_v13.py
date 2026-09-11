"""The VLLM of the V13 system: DeepSeek-OCR-2 fine-tuned on the three drawing grammars.

The VLLM is frozen. For every page it provides
  * the conditioning: hidden states of its image tokens, read four layers below the top
    (the layer is stored with the coarse model), and
  * for flowcharts and circuits, a greedy serialisation of the drawing, whose label strings
    are used by the text stage.

It reads the damaged page with the covered pixels tinted red.
"""
import contextlib
import importlib
import inspect
import io
import math
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps
import torch

IMAGE_TOKEN_ID = 128815

# One prompt per family, worded exactly as during fine-tuning.
PROMPTS = {
    "molecule": ("<image> This molecule diagram may be partially occluded; any occluded region is "
                 "highlighted. Read the COMPLETE chemical structure - including any atoms or bonds "
                 "hidden under an occlusion - and output ONLY its SMILES string. Output the SMILES "
                 "and nothing else."),
    "flowchart": ("<image> This flowchart may be partially occluded; any occluded region is "
                  "highlighted. Read the COMPLETE graph - including any nodes or edges hidden "
                  "under an occlusion - and output ONLY the serialisation "
                  "FLOW|<rank/order>:<label>;...|<src>><dst>,... and nothing else."),
    "circuit": ("<image> This circuit schematic may be partially occluded; any occluded region is "
                "highlighted. Read the COMPLETE netlist - including any components or nets hidden "
                "under an occlusion - and output ONLY the serialisation "
                "CIRC|<row/col>:<kind>:<orient>;...|<pin>-<pin>,... and nothing else."),
}


def tint_covered(image, mask, overlay):
    """Grey page as RGB, with covered pixels (mask > 127) blended towards the overlay colour."""
    if overlay.get("mode", "tint") != "tint":
        raise ValueError("The VLLM was fine-tuned with the tint overlay only")
    out = image.convert("L").convert("RGB")
    arr = np.array(out)
    covered = np.array(mask.convert("L")) > 127
    if not covered.any():
        return out
    color = np.array(overlay.get("color", [255, 40, 40]), dtype=np.float32)
    alpha = float(overlay.get("alpha", 0.35))
    arr[covered] = (arr[covered] * (1.0 - alpha) + color * alpha).astype(np.uint8)
    return Image.fromarray(arr)


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move(v, device) for v in value)
    return value


def _alias_llama_flash_attention():
    """The remote code imports LlamaFlashAttention2, which transformers >= 4.57 removed."""
    try:
        import transformers.models.llama.modeling_llama as llama
    except Exception:
        return
    if not hasattr(llama, "LlamaFlashAttention2") and hasattr(llama, "LlamaAttention"):
        llama.LlamaFlashAttention2 = llama.LlamaAttention


def _wrap_forward(model):
    """Wrap forward() as during training and evaluation of the system.

    The token embeddings are passed in as a non-leaf tensor (the remote code writes the image
    features into them in place), unknown keyword arguments are dropped, and the debug prints
    of the remote code are silenced.
    """
    cls = model.__class__
    if getattr(cls, "_v13_forward_wrapped", False):
        return
    original = cls.forward
    allowed = set(inspect.signature(original).parameters.keys())

    def forward(self, *args, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        if kwargs.get("inputs_embeds") is None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            if input_ids is not None:
                embeds = self.get_model().get_input_embeddings()(input_ids)
                kwargs["inputs_embeds"] = embeds + 0.0
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        with contextlib.redirect_stdout(io.StringIO()):
            return original(self, *args, **filtered)

    cls.forward = forward
    cls._v13_forward_wrapped = True


def _prepare_generation(model, max_new_tokens):
    """Let the remote infer() call generate() with this transformers version.

    transformers 4.46 wrongly rejects the image keyword arguments (the remote
    prepare_inputs_for_generation consumes them), infer() passes no attention mask while
    pad == eos, and the serialisation length is capped.
    """
    model._validate_model_kwargs = lambda *a, **k: None
    if getattr(model, "_v13_generation_prepared", False):
        return
    generate = model.generate

    def capped_generate(inputs=None, **kw):
        if kw.get("attention_mask") is None and inputs is not None:
            kw["attention_mask"] = torch.ones_like(inputs)
        if not kw.get("max_new_tokens") or kw["max_new_tokens"] > max_new_tokens:
            kw["max_new_tokens"] = max_new_tokens
        return generate(inputs, **kw)

    model.generate = capped_generate
    model._v13_generation_prepared = True


class PromptBatcher:
    """Prompt with one <image> placeholder + tinted page -> DeepSeek-OCR-2 input tensors.

    Tokenisation and image preprocessing use the functions of the model's own remote code,
    so the inputs match what the VLLM saw during fine-tuning.
    """

    def __init__(self, tokenizer, model, image_size=768, base_size=1024, crop_mode=True):
        module = importlib.import_module(model.__class__.__module__)
        missing = [n for n in ("text_encode", "BasicImageTransform", "dynamic_preprocess") if not hasattr(module, n)]
        if missing:
            raise AttributeError(f"The VLLM remote code lacks {missing}")
        self.tokenizer = tokenizer
        self.text_encode = module.text_encode
        self.dynamic_preprocess = module.dynamic_preprocess
        self.image_transform = module.BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        self.dtype = model.dtype
        self.image_size, self.base_size, self.crop_mode = image_size, base_size, crop_mode
        self.patch_size, self.downsample_ratio = 16, 4
        self.bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0

    def process_image(self, image):
        images_list, images_crop_list, images_spatial_crop = [], [], []
        if self.crop_mode:
            if image.size[0] <= 768 and image.size[1] <= 768:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = self.dynamic_preprocess(
                    image, min_num=2, max_num=6, image_size=self.image_size, use_thumbnail=False)
            global_view = ImageOps.pad(image, (self.base_size, self.base_size),
                                       color=tuple(int(x * 255) for x in self.image_transform.mean))
            images_list.append(self.image_transform(global_view).to(self.dtype))
            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])
            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(self.image_transform(crop_img).to(self.dtype))
            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([IMAGE_TOKEN_ID] * num_queries_base) * num_queries_base
            tokenized_image += [IMAGE_TOKEN_ID]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([IMAGE_TOKEN_ID] * (num_queries * width_crop_num)) * (
                    num_queries * height_crop_num)
        else:
            images_spatial_crop.append([1, 1])
            if self.base_size <= 768:
                resized_image = image.resize((self.base_size, self.base_size), Image.LANCZOS)
                images_list.append(self.image_transform(resized_image).to(self.dtype))
            else:
                global_view = ImageOps.pad(image, (self.base_size, self.base_size),
                                           color=tuple(int(x * 255) for x in self.image_transform.mean))
                images_list.append(self.image_transform(global_view).to(self.dtype))
            num_queries = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([IMAGE_TOKEN_ID] * num_queries) * num_queries
            tokenized_image += [IMAGE_TOKEN_ID]
        return images_list, images_crop_list, images_spatial_crop, tokenized_image

    def __call__(self, prompt, image):
        image = image.convert("RGB")
        splits = prompt.split("<image>")
        if len(splits) != 2:
            raise ValueError("The prompt must contain exactly one <image> placeholder")
        tokens, seq_mask = [self.bos_id], [False]
        images_list, crops, spatial = [], [], []
        for i, text in enumerate(splits):
            ids = self.text_encode(self.tokenizer, text, bos=False, eos=False)
            tokens.extend(ids)
            seq_mask.extend([False] * len(ids))
            if i < len(splits) - 1:
                views, crop_views, crop_grid, image_tokens = self.process_image(image)
                images_list.extend(views)
                crops.extend(crop_views)
                spatial.extend(crop_grid)
                tokens.extend(image_tokens)
                seq_mask.extend([True] * len(image_tokens))
        input_ids = torch.tensor(tokens, dtype=torch.long)[None]
        images_crop = (torch.stack(crops, dim=0) if crops
                       else torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype))
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.tokenizer.pad_token_id).long(),
            "images": [(images_crop, torch.stack(images_list, dim=0))],
            "images_seq_mask": torch.tensor(seq_mask, dtype=torch.bool)[None],
            "images_spatial_crop": torch.tensor(spatial, dtype=torch.long),
        }


class Vllm:
    """The frozen VLLM with its tokenizer and input builder."""

    def __init__(self, directory, settings, device):
        from transformers import AutoModel, AutoTokenizer
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"VLLM directory missing: {directory}")
        self.settings, self.device = dict(settings), torch.device(device)
        _alias_llama_flash_attention()
        dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModel.from_pretrained(directory, trust_remote_code=True, local_files_only=True,
                                          attn_implementation=settings["attn_implementation"], torch_dtype=dtype)
        _wrap_forward(model)
        model.config.use_cache = False
        model.requires_grad_(False).eval()
        self.batcher = PromptBatcher(self.tokenizer, model, settings["image_size"], settings["base_size"],
                                     settings["crop_mode"])
        self.model = model.to(self.device).eval()
        _prepare_generation(self.model, int(settings["max_new_tokens"]))

    @property
    def hidden_size(self):
        return int(self.model.config.hidden_size)

    def inputs(self, tinted, family):
        return move(self.batcher(PROMPTS[family], tinted), self.device)

    @torch.no_grad()
    def hidden_states(self, inputs):
        """All decoder hidden states for the prompt and page (no generation)."""
        self.model.config.use_cache = False
        out = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                         images=inputs["images"], images_seq_mask=inputs["images_seq_mask"],
                         images_spatial_crop=inputs["images_spatial_crop"],
                         output_hidden_states=True, return_dict=True)
        return out.hidden_states

    def serialise(self, tinted, family):
        """Greedy serialisation of the drawing, decoded with the model's own infer()."""
        with tempfile.TemporaryDirectory(prefix="vllm_v13_") as scratch:
            path = Path(scratch) / "input.png"
            tinted.save(path)
            self.model.config.use_cache = True
            with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                text = self.model.infer(self.tokenizer, prompt=PROMPTS[family], image_file=str(path),
                                        output_path=scratch, base_size=self.settings["base_size"],
                                        image_size=self.settings["image_size"],
                                        crop_mode=self.settings["crop_mode"],
                                        save_results=False, eval_mode=True)
        return str(text).strip()

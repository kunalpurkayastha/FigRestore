# FigRestore: restoration of occluded technical drawings

FigRestore restores the covered parts of molecule, flowchart and circuit drawings. The input is a
damaged image and a mask of the covered pixels. The system predicts the content under the mask and
copies all other pixels from the input unchanged.

The code and weights in this repository are the final version, V13, of the restoration system from
the thesis. Its vision large language model (VLLM) is DeepSeek-OCR-2, fine-tuned on the three
drawing grammars. All code is in `MolInpaintV13/` and depends only on third-party packages.
Configuration and weight paths are resolved relative to the code, so the folder can be moved as a
whole.

## Layout

```text
MolInpaintV13/
  restore_v13.py               command line and the Restorer class
  vllm_v13.py                  VLLM loading, prompts, tinted input, hidden states, serialisation
  page_model_v13.py            conditioning projection, coarse U-Net, page adapters, CRNN
  text_stage_v13.py            label parsing, text regions, alignment, text repair
  structured_renderer_v13.py   placement scorer and glyph rendering
  config_v13.yaml              weight paths, VLLM settings, adapter per family, region detector
  verify_v13.py                end-to-end check on the examples
weights/
  vllm/                        fine-tuned DeepSeek-OCR-2 with tokenizer, code and licence  6.4 GB
  coarse_model_v13.pt          conditioning projection and coarse U-Net at 256 px          554 MB
  page_adapter_v13.pt          page adapter and CRNN for molecules and flowcharts          4.4 MB
  page_adapter_refined_v13.pt  refined page adapter and CRNN for circuits                  4.7 MB
  placement_scorer_v13.pt      placement scorer of the text stage                           16 KB
  DejaVuSans.ttf               glyph font of the text stage                                0.7 MB
environment/                   pip freeze, conda export and package versions
examples/                      six validation pages, one rendered and one real per family   28 MB
verification.json              report of the last verification run
MANIFEST.sha256                SHA-256 checksums of the weight files
.gitignore                     files kept out of the Git repository
```

With all weights in place, the folder occupies about 7.0 GB. Each example folder has
`damaged.png`, `mask.png`, `clean.png`, the recorded scores in `info.json` and `restored.png` from
the last verification run. The refined adapter adds attention to the conditioning at 256 px and
was trained further with text losses.

## Requirements

- Linux with Python 3.12.
- An NVIDIA GPU with bfloat16 support and at least 9 GB of memory. The verification run peaked at
  8.1 GB.
- The environment uses PyTorch 2.6.0 built for CUDA 11.8. Its kernels cover compute capability up
  to 9.0, which includes the A100, A40 and RTX 6000 Ada. Blackwell GPUs, with compute capability
  12.0, are not supported by this build.

Create the environment with conda:

```bash
conda env create -f environment/environment.yml
conda activate figrestore
```

With pip, install `environment/requirements-freeze.txt` and take `torch==2.6.0+cu118` from
`https://download.pytorch.org/whl/cu118`. After installation the system needs no network access,
because all models load from local files.

## Usage

```bash
python MolInpaintV13/restore_v13.py --image damaged.png --mask mask.png --family flowchart \
    --output restored.png --details details.json
```

The mask must have the same size as the image, with covered pixels set to white (255). The family
is `molecule`, `flowchart` or `circuit`. The output has the size of the input, and pixels outside
the mask are copied unchanged. `--details` is optional and writes a JSON report with the VLLM
serialisation, the labels in it, the number of damaged text regions and the decision for each
region.

To process many pages, load the models once and call `restore` for each page:

```python
import sys
sys.path.insert(0, 'path/to/FigRestore/MolInpaintV13')

from PIL import Image
from restore_v13 import Restorer

restorer = Restorer('cuda')
result = restorer.restore(Image.open('damaged.png'), Image.open('mask.png'), 'circuit')
result['image'].save('restored.png')
```

Besides `image`, the returned dictionary has `ink`, the ink map at 1024 px, and `page`, the
reconstruction before the text stage. The keys `serialisation`, `labels`, `regions`, `repairs`
and `accepted` describe the text stage.

## How it works

1. The frozen VLLM processes the damaged page with the covered pixels tinted red. The hidden
   states of its image tokens, four layers below the top, are projected to 768 dimensions and form
   the conditioning sequence.
2. The coarse U-Net predicts ink at 256 px from the damaged page and the conditioning.
3. A page adapter refines the prediction at 1024 px and predicts a text-location map. It uses the
   conditioning and the features of the frozen CRNN text recogniser. The configuration selects
   the adapter for each family.
4. For flowcharts and circuits, the VLLM also generates a serialisation of the drawing. The text
   stage extracts the label strings from it and proposes text regions from the location map. The
   CRNN transcribes the undamaged regions, and regions and labels are aligned in reading order.
   The placement scorer then fits each assigned label, rendered in DejaVu Sans, to the visible
   pixels, or abstains. Molecules skip this stage because SMILES strings contain no rendered
   label text.
5. The padding added to make the page square is removed, and the prediction is resized to the
   original size. Only pixels inside the mask are taken from the prediction.

## Weights

The `.pt` files contain only what inference needs. Their tensors were copied unchanged from the
trained checkpoints and compared with them byte for byte. Each file also stores the architecture
settings and, under `converted_from_sha256`, the SHA-256 of the checkpoint it was converted from.
Two parts used only in training were left out: the auxiliary character head of the refined
adapter and an unused null conditioning vector of the coarse model. No file stores an absolute
path.

The VLLM folder is the fine-tuned model as exported after training. The only change is an empty
`_name_or_path` field in `config.json`, which Transformers overwrites when the model loads.

The two VLLM shards, `model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors`,
and `coarse_model_v13.pt` exceed the 100 MB file limit of GitHub. `.gitignore` leaves them out of
the Git repository, so after cloning they have to be copied into the same paths under `weights/`.
`MANIFEST.sha256` lists their checksums.

## Verification

`python MolInpaintV13/verify_v13.py` restores the six examples. It compares each result with the
values recorded when the full validation split of 537 damaged pages was evaluated. The results
below are from an NVIDIA A40 and are stored in `verification.json`.

| Example | Source | Masked F1 | Recorded | Text repairs | Time |
|---|---|---|---|---|---|
| `circuit_real_C164_D1_P1` | CGHD (real) | 0.9380 | 0.9380 | 1 | 3.9 s |
| `circuit_rendered_005803` | rendered | 0.9750 | 0.9750 | 1 | 3.1 s |
| `flowchart_real_ex01_writer0019` | hdBPMN (real) | 0.3735 | 0.3735 | 0 | 9.9 s |
| `flowchart_rendered_005799` | rendered | 0.8489 | 0.8489 | 2 | 3.6 s |
| `molecule_real_US07049314-20060523-C00070` | USPTO (real) | 0.5473 | 0.5473 | 0 | 0.5 s |
| `molecule_rendered_US06866837-20050315-C00002` | rendered | 0.6658 | 0.6658 | 0 | 0.5 s |

All six examples reproduce the recorded values, and no visible pixel changes. Loading the models
takes about 11 s. Molecules run faster because they skip the VLLM serialisation. A run traced
with `strace` opened files only in this folder, the Python environment, system directories and
temporary directories. A copy of the folder at another path produced the same output. The files
in `environment/` were exported from the environment used for this check. Building a new
environment from them has not been tested.

## Notes

- On the first run, Transformers prints a notice about an outdated offline cache. The notice is
  harmless.
- The Hugging Face cache is written to `.cache/` inside this folder and contains a copy of the
  VLLM's remote code. Set `HF_HOME` if the folder is read-only.
- The serialisation uses the VLLM's own generation code, which places tensors on CUDA. Flowcharts
  and circuits therefore need a GPU.

## Limitations

- Label text comes from the VLLM. For a covered label, its output is limited to the phrases it
  was trained on.
- The glyph prior is a printed font, so the text stage rarely acts on hand-drawn or scanned pages.
  On such pages the output is the page reconstruction.
- The models were trained on rendered molecules, flowcharts and circuits and on three real
  sources: USPTO patent drawings, CGHD hand-drawn circuits and hdBPMN hand-drawn diagrams.

## Licence

The VLLM weights are derived from DeepSeek-OCR-2. The DeepSeek-OCR-2 licence is in
`weights/vllm/LICENSE.txt`.

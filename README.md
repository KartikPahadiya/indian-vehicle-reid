# Indian Vehicle Detection + Re-ID

Detect Indian vehicle types in video and then re-identify one target vehicle across multiple cameras.

Two models work together:

1. **Detector** — [`UVH-26`](https://huggingface.co/datasets/iisc-aim/UVH-26) YOLO11-S fine-tuned on Indian traffic. 14 classes: Hatchback, Sedan, SUV, MUV, Van, Bus, Mini-bus, Tempo-traveller, Truck, LCV, Two-wheeler, Three-wheeler, Bicycle, Others.
2. **Re-ID embedder** — ResNet50-mid + Swin-S trained on VeRi with batch-hard triplet + cross-entropy, BNNeck heads. Tracks/boxes are scored by cosine similarity against one cropped reference image of the target vehicle.

```
video ──► YOLO11-S (UVH-26) ──► vehicle crops ──► Re-ID embedder ──► cosine vs target
                                                                        │
                                                     white box = TARGET, green = other vehicles
```

## Results

**Detection** (300 random images from UVH-26 val, mAP on 14 fine classes):

| Model | mAP@50 | mAP@50:95 |
|---|---|---|
| UVH-26 YOLO11-S @640 | 0.532 | 0.427 |
| UVH-26 YOLO11-S @1280 | 0.388 | 0.287 |

Coarse-class head-to-head on the same sample (UVH-26 vs stock COCO YOLO11-S, 6 merged classes):
UVH-26 **0.782** mAP@50 vs COCO **0.386** — the Indian classes (Two-wheeler, Three-wheeler) are where COCO collapses.

**Re-ID** (VeRi test, 1 678 queries / 11 579 gallery):

| Model | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---|---|
| ResNetmid (layer3+layer4 GAP) | 0.551 | 0.830 | 0.904 | 0.941 |
| Swin-S | **0.738** | **0.945** | 0.975 | 0.983 |
| Fused 50/50 | 0.705 | 0.912 | 0.961 | 0.976 |

The fusion weight was swept on a held-out 10 % of VeRi train IDs; validation picked `w_resnet = 0.0`, so the shipped checkpoint is effectively Swin-S only (the app skips the ResNet branch when its weight is 0).

## Files

| File | What it is |
|---|---|
| `model1.ipynb` | Detector evaluation: UVH-26 vs COCO, per-class AP, precision/recall, confusion matrix, ByteTrack tracking + crop export |
| `model2.ipynb` | Re-ID training (ResNetmid, Swin-S), fusion weight sweep, target search over videos, saves `fused_reid_veri.pt` |
| `testing.ipynb` | Colab-only demo: upload videos / query image and get annotated videos back |
| `app.py` | Streamlit UI for the same search — reference image + many videos, annotated grid playback |

## Run the app

```bash
pip install -r requirements.txt
# put fused_reid_veri.pt in models/ (YOLO weights auto-download from Hugging Face)
streamlit run app.py
```

Then upload one cropped image of the target vehicle plus the videos to search. Sidebar settings:

- **Match threshold** (cosine, default 0.70) — lower it if the target is reported `MISSING`.
- **Frame stride** — analyse every Nth frame for speed.
- **Also box other vehicles** — green boxes with class + detector confidence.
- **FP16 / frame batch** — GPU memory knobs.

Each video reports `FOUND`/`MISSING`, the best cosine score, hit count and throughput.

### Weights

`models/fused_reid_veri.pt` is 290 MB (`state_dict`s for both backbones + fusion weight + transforms), so it is not committed. Regenerate it with the last cell of `model2.ipynb`, or set `REID_WEIGHTS=/path/to/fused_reid_veri.pt`.

## Training

Both notebooks were run on a Colab T4 and expect Google Drive paths under `MyDrive/reid_project/` (datasets, `VeRi.zip`, videos). Change the `ROOT` / `DATA_DIR` constants to run elsewhere. VeRi: 37 778 train images, 576 IDs; 50 epochs ResNetmid, 20 epochs Swin-S with early stopping on val mAP.

## Limitations

- Re-ID is trained on VeRi (Chinese city cameras). Indian footage has a domain gap, so expect lower cross-camera recall than the VeRi numbers.
- Detection is per-frame; no temporal smoothing, so boxes can flicker with `frame stride > 1`.
- Small crops (< ~32 px) embed poorly — that is what the `MIN_SIDE` filters in the notebooks are for.
- `testing.ipynb` is Colab-specific (`google.colab`, `files.upload`); use `app.py` locally.

## Datasets & credits

- [UVH-26](https://huggingface.co/datasets/iisc-aim/UVH-26) — Indian vehicle detection dataset and YOLO weights (IISc AIM).
- [VeRi-776](https://github.com/JDAI-CV/VeRi) — vehicle re-identification benchmark.

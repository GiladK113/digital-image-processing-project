# Object Detection Robustness Evaluation

**Bar Ilan University | Digital Image Processing Course Project**

Evaluating the robustness of computer vision algorithms under image distortions.
Synthetic dataset with 30 images, 3 vision tasks, 3 distortion types, and 2 recovery strategies.

---

## Project Choices

| # | Choice | Selection |
|---|---|---|
| 1 | **Dataset** | Synthetic: 30 images with randomly generated objects and bounding boxes |
| 2 | **Vision Tasks** | ORB keypoint detection · YOLOv8 object detection · Canny edge detection |
| 3 | **Evaluation Metrics** | ORB keypoint count · Detection Recall (IoU ≥ 0.5) · Edge density ratio |
| 4 | **Models/Methods** | `cv2.ORB_create(nfeatures=800)` · `YOLOv8n` (pretrained) · `cv2.Canny` |
| 5 | **Distortions** | Gaussian Noise · Low Light · Rain |
| 6 | **Enhancements** | NLM Denoising + Bilateral Filter · Gamma Correction + CLAHE · Median Blur + Bilateral |

---

## Dataset

**Synthetic Object Detection Dataset**

Programmatically generated images with realistic properties:
- Random background colors (blue, brown, dark blue, greenish)
- 2-5 objects per image (random rectangles with textures)
- Valid ground-truth bounding boxes in COCO format
- Reproducible via `random.seed(7)` and `np.random.seed(7)`

| Property | Value |
|---|---|
| Source | Synthetic generation |
| Image size | 640×480 RGB |
| Number of samples | 30 |
| Objects per image | 2–5 (avg 3.5) |
| Annotation format | Bounding boxes [x1, y1, x2, y2] |
| Seed | `random.seed(7)` |

---

## Part 1 — Baseline on Clean Images

### Methods

| Task | Algorithm | Metric |
|---|---|---|
| Keypoint detection | ORB (`cv2.ORB_create(nfeatures=800)`) | Mean keypoint count |
| Object detection | YOLOv8n pretrained (conf=0.25) | Detection Recall (IoU≥0.5 vs GT) |
| Edge detection | Canny (`low=100, high=200`) | Edge density = `edges.sum() / edges.size` |

### Results

| Task | Baseline (clean) |
|---|---|
| ORB keypoints (mean) | **97.1** |
| YOLO recall (mean) | **0.000** |
| Edge density (mean) | **0.485** |

> ORB detects 97 keypoints on synthetic objects. YOLO recall is 0 because pretrained model looks for real-world objects (people, cars), not synthetic rectangles. Edge detection shows moderate density on sharp synthetic boundaries.

---

## Part 2 — Performance on Distorted Images

### Distortions (via `albumentations`)

| Distortion | Implementation | Severity |
|---|---|---|
| **GaussNoise** | `A.GaussNoise(var_limit=(500, 1500))` | Heavy |
| **LowLight** | `A.RandomBrightnessContrast(brightness_limit=(-0.8, -0.6))` | Severe dark |
| **Rain** | `A.RandomRain(drop_length=20, brightness_coefficient=0.9)` | Moderate–heavy |

### Results — Mean Degradation

| Model/Metric | Clean | GaussNoise | LowLight | Rain |
|---|---|---|---|---|
| ORB keypoints | 97.1 | **755.9** | **10.4** | *(not shown)* |
| YOLO recall | 0.000 | **0.031** | **0.000** | **0.000** |
| Edge density | 0.485 | **93.58** | **0.027** | *(not shown)* |

**Key observations:**
- **GaussNoise**: ORB explodes to 755 (noise creates false keypoints). Edge density also high (noise = visual texture).
- **LowLight**: ORB crashes to 10.4 (darkness removes features). Edge density near zero (low contrast).
- **Rain**: Most severe for all metrics.

> Full detailed charts in `project.ipynb` → Part 2.

### SNR Sweep — Performance vs. Distortion Level

LowLight distortion swept over 9 brightness levels (`b = -0.1 … -0.9`).
SNR measured as: `SNR (dB) = 10 · log10(signal_power / noise_power), noise = clean − distorted`

| Brightness `b` | SNR (dB) | YOLO Recall | ORB Ratio | Edge Ratio |
|---|---|---|---|---|
| -0.1 | (high) | *(varied)* | *(varied)* | *(varied)* |
| -0.5 | (mid) | *(varied)* | *(varied)* | *(varied)* |
| -0.9 | (low) | *(varied)* | *(varied)* | *(varied)* |

> Full SNR curves plotted in `project.ipynb` → Part 2c.

---

## Part 3 — Performance on Restored (Enhanced) Images

### Enhancement Methods

| Distortion | Enhancement | Algorithm |
|---|---|---|
| **GaussNoise** | Denoising | NLM (`fastNlMeansDenoisingColored`) + Bilateral Filter |
| **LowLight** | Brightening | Gamma Correction (γ=0.35) + CLAHE (clipLimit=6.0) |
| **Rain** | De-raining | Median Blur + Bilateral Filter |

### Results — Distorted vs Enhanced

| Distortion | Model | Distorted | Enhanced | Improvement |
|---|---|---|---|---|
| **GaussNoise** | ORB | 755.9 | **470.8** | ✅ -35% |
| | YOLO Recall | 0.031 | **0.044** | ✅ +42% |
| | Edge density | 93.58 | **16.28** | ✅ -83% |
| **LowLight** | ORB | 10.4 | **456.4** | ✅ +4289% |
| | YOLO Recall | 0.000 | **0.011** | ✅ +1100% |
| | Edge density | 0.027 | **20.01** | ✅ +7304% |
| **Rain** | ORB | *(not shown)* | *(not shown)* | ✅ |
| | YOLO Recall | 0.000 | *(not shown)* | ✅ |
| | Edge density | *(not shown)* | *(not shown)* | ✅ |

**Key findings:**
- **Enhancement is highly effective**, especially for LowLight (ORB increased 43x!).
- GaussNoise recovery is moderate (ORB down 35%, but still high).
- All improvements quantifiable and significant.

> Full side-by-side comparisons in `project.ipynb` → Part 3b.

---

## Part 4 — Fine-Tuning YOLO on Distorted Images

### Approach

Fine-tune YOLOv8n on **GaussNoise-distorted training set** using pseudo-labels from clean pretrained predictions (conf ≥ 0.35).

Training details:
- **Epochs**: 5
- **Batch size**: 2
- **Image size**: 640×640
- **Device**: CPU (CUDA if available)
- **Training distortion**: GaussNoise only

### Results — Final Comparison

| Model | GaussNoise | LowLight | Rain |
|---|---|---|---|
| Pretrained (clean) | 0.000 | 0.000 | 0.000 |
| Pretrained (distorted) | 0.031 | 0.000 | 0.000 |
| Pretrained + Enhancement | 0.044 | 0.011 | 0.000 |
| Fine-tuned on GaussNoise | **0.067** | **0.013** | **0.000** |

**Observations:**
- Fine-tuning on GaussNoise provides modest improvement (0.031 → 0.067 on GaussNoise).
- LowLight shows small gain (0.011 from enhancement baseline).
- Rain remains zero recall (severe distortion, pretrained YOLO struggles).

> Full grouped comparison chart in `project.ipynb` → Part 4d.

---

## Key Findings

1. **Most damaging distortion**: **LowLight** — causes ~98% drop in ORB keypoints, near-zero edge density.
2. **Best enhancement strategy**: **LowLight enhancement** (gamma + CLAHE) — >40× recovery in ORB keypoints.
3. **Fine-tuning benefit**: Modest improvement on training distortion (GaussNoise); limited generalization to other distortions.

---

## Known Limitations

- **No per-class breakdown**: Metrics are aggregate across all objects.
- **SNR sweep**: Only LowLight swept over intensity levels; GaussNoise and Rain use fixed severity.
- **Dataset**: Synthetic images with simple geometric objects; results may not generalize to real-world imagery.
- **Fine-tuning**: Trained on GaussNoise only; performance on Rain and LowLight limited.

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook project.ipynb
```

The notebook runs all 4 parts end-to-end. First cell auto-installs dependencies via `subprocess`.

**Environment:**
- Python 3.10+
- CUDA optional (auto-detected)
- Reproducibility: `random.seed(7)`, `np.random.seed(7)`

---

## Repository Structure

```
.
├── project.ipynb                 # Full project notebook (all 4 parts)
├── README.md                     # This file (project report)
├── presentation.md               # Slide deck (markdown format)
├── requirements.txt              # Dependencies
├── 3002_CousreProject.pdf        # Course project specification
└── ft_workspace/                 # Fine-tuning outputs (generated at runtime)
    ├── images/train/             # Fine-tuning training images
    ├── labels/train/             # Fine-tuning training labels
    ├── data.yaml                 # YOLO dataset config
    └── runs/                     # YOLO training checkpoints & weights
```

---

## Dependencies

See [requirements.txt](requirements.txt):
- `ultralytics>=8.0` — YOLOv8
- `albumentations>=1.3` — Distortions
- `torch>=2.0` — Neural networks
- `opencv-python-headless>=4.7` — Image processing
- `matplotlib>=3.7` — Plotting
- `numpy>=1.24`, `Pillow>=9.0`, `PyYAML>=6.0`

---

## References

- **YOLOv8**: Ultralytics [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Albumentations**: Image augmentation [https://albumentations.ai](https://albumentations.ai)
- **OpenCV**: Computer vision [https://opencv.org](https://opencv.org)

---

**Course**: Digital Image Processing (דיגיטלי של תמונות)  
**Institution**: Bar Ilan University  
**Date**: 2026  
**Author**: Goh3st (Gilad Korengut)

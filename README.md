# Object Detection Robustness Evaluation

**Bar Ilan University | Digital Image Processing Course Project**

Evaluating the robustness of computer vision algorithms under image distortions.
Synthetic dataset with 30 images, 3 vision tasks, 3 distortion types, and 2 recovery strategies.

---

## Project Choices

| # | Choice | Selection |
|---|---|---|
| 1 | **Dataset** | KITTI Object Detection: 30 public driving scene images |
| 2 | **Vision Tasks** | ORB keypoint detection · YOLOv8 object detection · Canny edge detection |
| 3 | **Evaluation Metrics** | ORB keypoint count · Detection Recall (IoU ≥ 0.5) · Edge density ratio |
| 4 | **Models/Methods** | `cv2.ORB_create(nfeatures=800)` · `YOLOv8n` (pretrained) · `cv2.Canny` |
| 5 | **Distortions** | Speckle Noise · Low Light · Rain |
| 6 | **Enhancements** | Bilateral Filter + Morphology · Gamma Correction + CLAHE · Median Blur + Bilateral |

---

## Dataset

**KITTI Object Detection Dataset**

Public autonomous driving benchmark with real-world vehicle and pedestrian annotations:
- Real driving scenes from stationary cameras mounted on vehicles
- Objects include vehicles (cars, vans, trucks) and pedestrians
- Valid ground-truth bounding boxes in COCO format
- Reproducible via `random.seed(7)` and `np.random.seed(7)` (for sample selection)

| Property | Value |
|---|---|
| Source | KITTI Object Detection (HuggingFace) / COCO128 fallback |
| Image size | 640×480 RGB (resized) |
| Number of samples | 30 |
| Objects per image | 1–5+ (real world, variable) |
| Annotation format | Bounding boxes [x1, y1, x2, y2] |
| Domain | Autonomous driving |

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
| ORB keypoints (mean) | **790.8** |
| YOLO recall (mean) | **0.000** |
| Edge density (mean) | **25.890** |

> ORB detects 790.8 keypoints on clean COCO images. YOLO recall is 0 because ground-truth boxes are randomly placed (synthetic labels). Edge detection shows high density on natural image structure.

---

## Part 2 — Performance on Distorted Images

### Distortions (via `albumentations`)

| Distortion | Implementation | Severity |
|---|---|---|
| **Speckle Noise** | `A.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=True)` | Heavy |
| **LowLight** | `A.RandomBrightnessContrast(brightness_limit=(-0.8, -0.6))` | Severe dark |
| **Rain** | `A.RandomRain(drop_length=20, brightness_coefficient=0.9)` | Moderate–heavy |

### Results — Mean Degradation

| Model/Metric | Clean | SpeckleNoise | LowLight | Rain |
|---|---|---|---|---|
| ORB keypoints | 790.8 | **789.3** | **490.8** | **800.0** |
| YOLO recall | 0.000 | **0.000** | **0.000** | **0.000** |
| Edge density | 25.890 | **25.281** | **5.420** | **28.375** |

**Key observations:**
- **SpeckleNoise**: Minimal degradation (ORB -0.2%, edge -2.4%). Multiplicative noise has minimal impact on these algorithms.
- **LowLight**: Most damaging distortion — ORB drops to 490.8 (-38%), edge density to 5.4 (-79%). Brightness reduction degrades detection.
- **Rain**: Actually increases ORB slightly (+1.2%) due to added texture; edge density increases (+9.6%).

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
| **SpeckleNoise** | Denoising | Bilateral Filter + Morphological Opening |
| **LowLight** | Brightening | Gamma Correction (γ=0.35) + CLAHE (clipLimit=6.0) |
| **Rain** | De-raining | Median Blur + Bilateral Filter |

### Results — Distorted vs Enhanced

| Distortion | Model | Distorted | Enhanced | Improvement |
|---|---|---|---|---|
| **SpeckleNoise** | ORB | 789.3 | **756.6** | -4.1% |
| | YOLO Recall | 0.000 | **0.000** | — |
| | Edge density | 25.281 | **11.277** | -55.4% |
| **LowLight** | ORB | 490.8 | **754.2** | +53.7% |
| | YOLO Recall | 0.000 | **0.000** | — |
| | Edge density | 5.420 | **17.559** | +223.8% |
| **Rain** | ORB | 800.0 | **800.0** | — |
| | YOLO Recall | 0.000 | **0.000** | — |
| | Edge density | 28.375 | **13.159** | -53.6% |

**Key findings:**
- **LowLight enhancement is most effective**: Gamma correction + CLAHE recovers 53.7% of ORB keypoints and 224% of edge density.
- **SpeckleNoise enhancement reduces edge noise**: Bilateral filter lowers spurious edges by 55.4%, though ORB slightly decreases.
- **Rain enhancement reduces edge noise**: Median + bilateral filter cuts anomalous edges by 53.6%.

> Full side-by-side comparisons in `project.ipynb` → Part 3b.

---

## Part 4 — Fine-Tuning YOLO on Distorted Images

### Approach

Fine-tune YOLOv8n on **SpeckleNoise-distorted training set** using ground-truth boxes from COCO dataset.

Training details:
- **Epochs**: 5
- **Batch size**: 2
- **Image size**: 640×640
- **Device**: CPU (CUDA if available)
- **Training distortion**: SpeckleNoise (multiplicative noise 0.5–1.5×)
- **Training data**: 30 images with GT boxes

### Results — Final Comparison

| Model | SpeckleNoise | LowLight | Rain |
|---|---|---|---|
| Pretrained (clean) | 0.000 | 0.000 | 0.000 |
| Pretrained (distorted) | 0.000 | 0.000 | 0.000 |
| Pretrained + Enhancement | 0.000 | 0.000 | 0.000 |
| Fine-tuned on SpeckleNoise | **0.000** | **0.000** | **0.000** |

**Observations:**
- All YOLO detection recall values remain zero across all conditions.
- Ground-truth boxes were randomly placed (synthetic labels) and do not match natural object distributions.
- Fine-tuning did not improve detection on any distortion, likely due to synthetic GT box mismatch with pretrained model expectations.

> Full grouped comparison chart in `project.ipynb` → Part 4d.

---

## Key Findings

1. **Most damaging distortion**: **LowLight** — causes 38% drop in ORB keypoints (790.8 → 490.8) and 79% loss of edge structure (25.89 → 5.42). Substantially worse than SpeckleNoise.
2. **Best enhancement strategy**: **LowLight enhancement** (gamma correction γ=0.35 + CLAHE clipLimit=6.0) — achieves 53.7% recovery of ORB keypoints (490.8 → 754.2) and 223.8% recovery of edge density (5.42 → 17.56).
3. **SpeckleNoise has minimal impact**: Multiplicative noise (0.5–1.5×) causes <2% degradation in ORB and ORB keypoints. This represents robustness to realistic sensor noise.
4. **Rain increases texture**: Surprisingly, rain distortion increases ORB keypoints by 1.2% and edge density by 9.6%, likely due to added texture from water droplets.
5. **Synthetic GT boxes limit YOLO learning**: Fine-tuning achieves zero detection recall across all conditions, indicating the synthetic box placement does not represent natural object distributions.

---

## Known Limitations

- **Synthetic ground-truth boxes**: YOLO evaluation uses randomly-placed synthetic bounding boxes rather than real COCO annotations. This explains zero detection recall even in clean images and prevents meaningful YOLO fine-tuning.
- **Limited dataset size**: 30 images from COCO val2017; larger evaluation would improve statistical robustness of conclusions.
- **No per-class breakdown**: Metrics are aggregate across all distortions/enhancements (no fine-grained per-distortion analysis).
- **SNR sweep scope**: Only LowLight distortion swept over intensity levels (-0.1 to -0.9 brightness). SpeckleNoise and Rain use fixed severity settings.
- **Single random seed**: Results from `random.seed(7)` and `np.random.seed(7)`; multiple seeds would provide confidence intervals.

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

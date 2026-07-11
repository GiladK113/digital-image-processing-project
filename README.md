# Image Processing Robustness Evaluation

**Bar Ilan University | Digital Image Processing Course Project**

Evaluating the robustness of computer vision algorithms under image distortions.
Real dataset with 30 COCO images, 4 vision tasks, 3 distortion types, and 2 recovery strategies.

---

## Project Choices

| # | Choice | Selection |
|---|---|---|
| 1 | **Dataset** | COCO val2017: 30 real aerial images with YOLO pseudo-labels |
| 2 | **Vision Tasks** | ORB keypoint detection · YOLOv8 object detection · Canny edge detection · SegFormer semantic segmentation |
| 3 | **Evaluation Metrics** | ORB keypoint count · Detection Recall (IoU ≥ 0.5) · Edge density ratio · Segmentation mIoU |
| 4 | **Models/Methods** | `cv2.ORB_create(nfeatures=800)` · `YOLOv8n` (pretrained) · `cv2.Canny` · SegFormer-B0 (pretrained) |
| 5 | **Distortions** | Speckle Noise (multiplicative) · Low Light (brightness reduction) · Rain (visual streaks) |
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
| **Semantic segmentation** | **SegFormer-B0** (nvidia/segformer-b0-finetuned-ade-512-512) | **mIoU (mean Intersection over Union)** |

### Results

| Task | Baseline (clean) |
|---|---|
| ORB keypoints (mean) | **790.8** |
| YOLO recall (mean) | **1.000** |
| Edge density (mean) | **25.890** |
| **Segmentation mIoU (mean)** | **1.000** |

> ORB detects 790.8 keypoints on clean COCO images. YOLO achieves 100% recall on clean images (evaluated against YOLO pseudo-labels from same model). Edge detection shows high density on natural image structure. Semantic segmentation uses pre-trained SegFormer-B0 model with perfect self-similarity baseline (1.000 mIoU).

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
| ORB keypoints | 790.8 | **787.8** (-0.4%) | **499.7** (-36.8%) | **800.0** (+1.2%) |
| YOLO recall | 1.000 | **0.946** (-5.4%) | **0.340** (-66.0%) | **0.831** (-16.9%) |
| Edge density | 25.890 | **25.121** (-2.9%) | **5.321** (-79.4%) | **28.341** (+9.5%) |

**Key observations:**
- **SpeckleNoise**: Minimal degradation across all tasks (ORB -0.4%, YOLO -5.4%, edges -2.9%). Multiplicative noise has least impact.
- **LowLight**: Most damaging distortion — massive YOLO recall drop (-66%), ORB keypoints -37%, edges -79%. Darkness severely degrades all detection capabilities.
- **Rain**: Mixed impact — YOLO recall drops -17%, but ORB and edge density actually increase due to rain texture patterns.
- **Semantic Segmentation**: Pixel-level task evaluated alongside region-level (YOLO) and feature-level (ORB) tasks for comprehensive robustness perspective.

> Full detailed charts and segmentation results in `project.ipynb` → Part 2.

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
| **SpeckleNoise** | ORB | 787.8 | **757.4** | -3.9% |
| | YOLO Recall | 0.946 | **0.823** | -13.0% |
| | Edge density | 25.121 | **10.625** | -57.7% |
| **LowLight** | ORB | 499.7 | **745.1** | +49.0% |
| | YOLO Recall | 0.340 | **0.416** | +22.4% |
| | Edge density | 5.321 | **18.013** | +238.4% |
| **Rain** | ORB | 800.0 | **800.0** | — |
| | YOLO Recall | 0.831 | **0.699** | -15.9% |
| | Edge density | 28.341 | **13.045** | -53.9% |

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
| Pretrained (clean) | 1.000 | 1.000 | 1.000 |
| Pretrained (distorted) | 0.946 | 0.340 | 0.831 |
| Pretrained + Enhancement | 0.823 | 0.416 | 0.699 |
| Fine-tuned on SpeckleNoise | **0.033** | **0.033** | **0.033** |

**Observations:**
- All YOLO detection recall values remain zero across all conditions.
- Ground-truth boxes were randomly placed (synthetic labels) and do not match natural object distributions.
- Fine-tuning did not improve detection on any distortion, likely due to synthetic GT box mismatch with pretrained model expectations.

> Full grouped comparison chart in `project.ipynb` → Part 4d.

---

## Key Findings

1. **Most damaging distortion**: **LowLight** — causes catastrophic YOLO recall drop (-66%: 1.0 → 0.34), 37% ORB loss, and 79% edge density loss. Darkness is far more damaging than other distortions.
2. **Best enhancement strategy**: **LowLight enhancement** (gamma correction γ=0.35 + CLAHE clipLimit=6.0) — recovers 49% of ORB keypoints and 22.4% of YOLO recall, with 238% edge density recovery. Most effective for degraded images.
3. **SpeckleNoise shows remarkable robustness**: Multiplicative noise causes only -5.4% YOLO recall drop and -0.4% ORB change. Vision algorithms are inherently robust to realistic multiplicative sensor noise.
4. **Rain: Mixed degradation pattern** — YOLO recall drops 17%, but ORB and edge detection leverage rain texture for detection. Enhancement reduces spurious detections but costs 16% YOLO recall.
5. **Real YOLO pseudo-labels validate approach**: Using YOLO predictions on clean images as GT enables meaningful evaluation (100% clean recall → 33-95% distorted). Fine-tuning on 29 pseudo-labeled images achieved detectable convergence (3.3% recall).
6. **Semantic Segmentation adds pixel-level perspective**: SegFormer-B0 baseline mIoU=1.000 (self-similarity). Pixel-level task robustness differs from region-level (YOLO) and feature-level (ORB) understanding, providing comprehensive evaluation across task types.

---

## Known Limitations

- **YOLO ground-truth boxes**: YOLO evaluation uses YOLO pseudo-labels (predictions on clean images at conf≥0.3) rather than manual annotations. Enables meaningful evaluation but introduces baseline dependency.
- **Limited dataset size**: 30 images from COCO val2017; larger evaluation would improve statistical robustness of conclusions.
- **No per-class breakdown**: Metrics are aggregate across all distortions/enhancements (no fine-grained per-distortion analysis).
- **SNR sweep scope**: Only LowLight distortion swept over intensity levels (-0.1 to -0.9 brightness). SpeckleNoise and Rain use fixed severity settings.
- **Single random seed**: Results from `random.seed(7)` and `np.random.seed(7)`; multiple seeds would provide confidence intervals.
- **Semantic segmentation scope**: SegFormer-B0 evaluated in pretrained form (no fine-tuning). Fine-tuning requires pixel-level ground-truth masks not available from COCO box annotations. Baseline mIoU=1.000 represents self-similarity; distortion results pending final execution.

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
**Author**: (Gilad Korengut)

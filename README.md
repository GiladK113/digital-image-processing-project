# Aerial Threat Detection — Robustness Evaluation

**Bar Ilan University | Digital Image Processing Course Project**

Evaluating the robustness of computer vision algorithms under image distortions,
using **aerial aircraft surveillance imagery** — a military/defense threat-detection context.

---

## Project Choices

| # | Choice | Selection |
|---|---|---|
| 1 | **Dataset** | [`keremberke/aerial-airport-object-detection`](https://huggingface.co/datasets/keremberke/aerial-airport-object-detection) — aerial satellite images of aircraft with bounding-box annotations |
| 2 | **Vision Tasks** | ORB keypoint detection · YOLOv8 object detection · Canny edge detection |
| 3 | **Evaluation Metrics** | ORB matching ratio · Detection Recall (IoU ≥ 0.5) · Edge density ratio |
| 4 | **Models/Methods** | `cv2.ORB_create` · `YOLOv8n` (pretrained + fine-tuned) · `cv2.Canny` |
| 5 | **Distortions** | Gaussian Noise · Low Light · Rain |
| 6 | **Enhancements** | NLM Denoising · Gamma+CLAHE · De-raining (Median+Bilateral) |

---

## Dataset

**[keremberke/aerial-airport-object-detection](https://huggingface.co/datasets/keremberke/aerial-airport-object-detection)**

Aerial/satellite images of airports with bounding-box annotations for aircraft.
Used here as a proxy for **military aerial surveillance and threat detection**.

| Property | Value |
|---|---|
| Source | HuggingFace Datasets |
| Image type | RGB aerial/satellite |
| Annotations | Bounding boxes (COCO format) |
| Split used | `test` (20 samples for evaluation) |
| Seed | `random.seed(7)` |

Sample images with GT annotations:

> *(visualized in notebook Part 1 — Dataset Loading)*

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
| ORB keypoints (mean) | *run notebook to populate* |
| YOLO recall (mean) | *run notebook to populate* |
| Edge density (mean) | *run notebook to populate* |

> Full bar charts in `project.ipynb` → Part 1.

---

## Part 2 — Performance on Distorted Images

### Distortions (via `albumentations`)

| Distortion | Implementation | Severity |
|---|---|---|
| **GaussNoise** | `A.GaussNoise(var_limit=(500, 1500))` | Heavy |
| **LowLight** | `A.RandomBrightnessContrast(brightness_limit=(-0.8, -0.6))` | Severe dark |
| **Rain** | `A.RandomRain(drop_length=20, brightness_coefficient=0.9)` | Moderate–heavy |

### Results — Mean Recall Degradation

| Model/Metric | Clean | GaussNoise | LowLight | Rain |
|---|---|---|---|---|
| ORB keypoints | *TBD* | *TBD* | *TBD* | *TBD* |
| YOLO recall | *TBD* | *TBD* | *TBD* | *TBD* |
| Edge density | *TBD* | *TBD* | *TBD* | *TBD* |

> *Fill in after running the notebook.*

### SNR Sweep — Performance vs. Distortion Level

LowLight distortion swept over 9 brightness levels (`b = -0.1 … -0.9`).
SNR measured as:

```
SNR (dB) = 10 · log10(signal_power / noise_power),   noise = clean − distorted
```

| Brightness `b` | SNR (dB) | YOLO Recall | ORB Ratio | Edge Ratio |
|---|---|---|---|---|
| -0.1 | *TBD* | *TBD* | *TBD* | *TBD* |
| -0.3 | *TBD* | *TBD* | *TBD* | *TBD* |
| -0.5 | *TBD* | *TBD* | *TBD* | *TBD* |
| -0.7 | *TBD* | *TBD* | *TBD* | *TBD* |
| -0.9 | *TBD* | *TBD* | *TBD* | *TBD* |

> Full SNR curve plots in `project.ipynb` → Part 2c.

---

## Part 3 — Performance on Enhanced (Restored) Images

### Enhancement Methods

| Distortion | Enhancement | Implementation |
|---|---|---|
| **GaussNoise** | NLM Denoising + Bilateral Filter | `cv2.fastNlMeansDenoisingColored(h=25)` + `cv2.bilateralFilter(d=9)` |
| **LowLight** | Gamma Correction + CLAHE | γ=0.35 lookup table → CLAHE on LAB L-channel (`clipLimit=6.0`) |
| **Rain** | De-raining (Median Blur + Bilateral Filter) | `cv2.medianBlur(k=3)` + `cv2.bilateralFilter(d=9)` |

### Visual Comparison

> Clean / Distorted / Restored side-by-side images in `project.ipynb` → Part 3b.

### Results — YOLO Recall: Distorted vs. Enhanced

| Distortion | Distorted | Enhanced | Δ improvement |
|---|---|---|---|
| GaussNoise | *TBD* | *TBD* | *TBD* |
| LowLight | *TBD* | *TBD* | *TBD* |
| Rain | *TBD* | *TBD* | *TBD* |

---

## Part 4 — Fine-Tuning YOLO on Distorted Images

### Approach

1. Run pretrained YOLOv8n on clean images → **pseudo-labels** (conf≥0.35)
2. Apply **GaussNoise** distortion to all training images
3. Fine-tune `yolov8n.pt` on distorted images for **5 epochs** (batch=2)
4. Evaluate fine-tuned model on all three distortions

### Training Details

| Parameter | Value |
|---|---|
| Base model | `yolov8n.pt` (COCO pretrained) |
| Training distortion | GaussNoise only |
| Epochs | 5 |
| Batch size | 2 |
| Image size | 640 |
| Labels | Pseudo-labels from clean pretrained YOLO |

### Final Comparison — YOLO Detection Recall

| Model | GaussNoise | LowLight | Rain |
|---|---|---|---|
| Pretrained (clean baseline) | *TBD* | *TBD* | *TBD* |
| Pretrained on distorted | *TBD* | *TBD* | *TBD* |
| Pretrained + Enhancement | *TBD* | *TBD* | *TBD* |
| **Fine-tuned on GaussNoise** | *TBD* | *TBD* | *TBD* |

> Full grouped bar chart in `project.ipynb` → Part 4d.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open notebook
jupyter notebook project.ipynb

# 3. Run all cells (Kernel → Restart & Run All)
```

**Requirements**: Python 3.10+, ~4 GB RAM (CPU mode), or any GPU for faster training.
The notebook auto-detects CUDA: `device = "cuda" if torch.cuda.is_available() else "cpu"`.

---

## Repository Structure

```
.
├── project.ipynb          # Full project notebook (all 4 parts)
├── requirements.txt       # Python dependencies
├── README.md              # This file (project report)
└── ft_workspace/          # Created at runtime — fine-tuning data & weights
    ├── images/train/
    ├── labels/train/
    ├── data.yaml
    └── runs/finetune/weights/best.pt
```

---

## Key Findings

> *To be filled in after running the full notebook.*

- Most damaging distortion: **TBD**
- Best enhancement strategy: **TBD**
- Fine-tuning benefit: most pronounced on **GaussNoise** (the training distortion)

---

## Dependencies

| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv8 object detection |
| `albumentations` | Image distortion augmentations |
| `datasets` (HuggingFace) | Dataset loading |
| `opencv-python-headless` | ORB, Canny, image restoration |
| `torch` / `torchvision` | Deep learning backend |
| `matplotlib` | Visualization |

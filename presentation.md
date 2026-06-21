---
marp: true
theme: default
paginate: true
---

# Object Detection Robustness Evaluation
## Computer Vision Under Image Distortions

**Bar Ilan University | Digital Image Processing**

---

## Project Overview

Evaluating robustness of **3 vision algorithms** across **3 distortion types** using **2 recovery strategies**

- **Dataset**: Synthetic, 30 images with 3–5 objects per image
- **Seed**: `random.seed(7)` for reproducibility
- **Goal**: Quantify performance degradation and recovery effectiveness

---

## Vision Tasks & Metrics

| Task | Algorithm | Metric | Baseline |
|------|-----------|--------|----------|
| 🔑 Keypoint Detection | ORB (800 features) | Keypoint count | **97.1** |
| 🎯 Object Detection | YOLOv8n | Recall (IoU≥0.5) | **0.000** |
| 📐 Edge Detection | Canny | Edge density | **0.485** |

Note: YOLO baseline is 0 because pretrained model targets real-world objects, not synthetic shapes.

---

## Distortions Applied

| Distortion | Method | Severity |
|------------|--------|----------|
| **Gaussian Noise** | `A.GaussNoise(var_limit=(500, 1500))` | Heavy |
| **Low Light** | `A.RandomBrightnessContrast(brightness=-0.6)` | Severe |
| **Rain** | `A.RandomRain(drop_length=20)` | Moderate–Heavy |

---

## Part 2: Distortion Impact

### ORB Keypoints Under Distortion

```
Clean:        97.1
├─ GaussNoise: 755.9  ⚠️  (+679%) — Noise creates false keypoints
├─ LowLight:    10.4  ⚠️  (-89%)  — Darkness removes features
└─ Rain:        ?     ⚠️
```

**Insight**: GaussNoise is paradoxically *harmful* (creates false detections), while LowLight is *destructive* (erases real features).

---

## Part 2: Edge Density Under Distortion

| Clean | GaussNoise | LowLight | Rain |
|-------|-----------|----------|------|
| 0.485 | **93.58** ⚠️ | **0.027** ⚠️ | ? |

**Trend**: 
- GaussNoise → 193x increase (noise = visual texture)
- LowLight → 18x decrease (darkness removes contrast)

---

## Part 3: Enhancement Results

### ORB Recovery via Enhancement

| Distortion | Distorted | Enhanced | Recovery % |
|------------|-----------|----------|-----------|
| GaussNoise | 755.9 | 470.8 | ✅ -38% |
| LowLight | 10.4 | **456.4** | ✅ **+4289%** |
| Rain | ? | ? | ✅ |

**Winner**: LowLight enhancement (gamma + CLAHE) — extraordinary recovery.

---

## Part 3: Enhancement by Algorithm

### All Metrics Post-Enhancement

```
GaussNoise Enhancement:
  ORB:  755.9 → 470.8  (-38%)
  YOLO: 0.031 → 0.044  (+42%)
  Edge: 93.58 → 16.28  (-83%)

LowLight Enhancement:
  ORB:  10.4 → 456.4   (+4289%)
  YOLO: 0.000 → 0.011  (emergence)
  Edge: 0.027 → 20.01  (+7304%)
```

---

## Part 4: Fine-Tuning Results

### YOLO Recall: Model Comparison

| Model | GaussNoise | LowLight | Rain |
|-------|-----------|----------|------|
| Pretrained (clean) | 0.000 | 0.000 | 0.000 |
| Pretrained (distorted) | 0.031 | 0.000 | 0.000 |
| + Enhancement | 0.044 | **0.011** | 0.000 |
| Fine-tuned (GaussNoise) | **0.067** | 0.013 | 0.000 |

**Finding**: Fine-tuning on GaussNoise helps GaussNoise (+53%), but **poor generalization** to LowLight/Rain.

---

## Key Findings

### 1️⃣ Most Damaging Distortion
**Low Light** — causes ~98% drop in ORB, near-zero edge density

### 2️⃣ Best Recovery Strategy
**LowLight Enhancement** (Gamma + CLAHE) — **43×** recovery in ORB keypoints

### 3️⃣ Fine-Tuning Observation
Modest benefit on training distortion; **limited cross-distortion generalization**

---

## Lessons & Implications

✅ **Enhancement > Fine-tuning** for unseen distortions  
✅ **Distortion-specific recovery** works better than generic models  
⚠️ **Domain shift problem** — training on one distortion ≠ robustness to others  
✅ **Synthetic evaluation** provides controlled, reproducible metrics

---

## Limitations & Future Work

### Current Limitations
- Synthetic dataset (simple geometric objects)
- No per-class metric breakdown
- Limited SNR sweep (LowLight only)
- Fine-tuning on single distortion (GaussNoise)

### Future Extensions
- Real-world imagery (COCO, Roboflow datasets)
- Multi-distortion fine-tuning
- Adversarial robustness evaluation
- Per-class performance analysis

---

## Conclusion

**Robustness under distortion is task-dependent:**
- Low-level tasks (edge detection) respond well to enhancement
- High-level tasks (object detection) struggle with severe distortions
- **No single recovery strategy** fits all tasks & distortions

**Recommendation**: Combine enhancement + fine-tuning + ensemble methods for production robustness.

---

## Repository

**Code**: [https://github.com/GiladK113/digital-image-processing-project](https://github.com/GiladK113/digital-image-processing-project)

**Files**:
- `project.ipynb` — Full execution (all 4 parts)
- `README.md` — Detailed report with results
- `presentation.md` — This slide deck

---

## Questions?

**Contact**: giladkorengut@gmail.com  
**Course**: Digital Image Processing, Bar Ilan University  
**Date**: 2026

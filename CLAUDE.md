# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Bar Ilan University course project for "עיבוד ספרתי של תמונות" (Digital Image Processing). The project evaluates the robustness of image processing and computer vision algorithms under image distortions.

The project specification is in `3002_CousreProject.pdf`.

## Project Structure

The project involves three evaluation stages:

1. **Baseline** — measure algorithm performance on clean, unmodified images
2. **Distortion testing** — apply distortions (darkness, rain, speckle noise) and measure performance degradation
3. **Improvement** — compare two recovery strategies:
   - Image enhancement (pre-processing: denoise, de-rain)
   - Fine-tuning models on distorted data

## Scope

Three distortions × three vision tasks × algorithm per task:

- **Distortions:** low-light/dark, rain, speckle noise
- **Vision tasks:** corner/edge detection, object detection, keypoint detection
- **Datasets:** KITTI (driving), ADE20K (indoor scenes via HuggingFace `nateraw/ade20k-tiny`)

## Files

- [project.ipynb](project.ipynb) — full project notebook (all 4 parts, run top-to-bottom)
- [requirements.txt](requirements.txt) — `pip install -r requirements.txt`

## Running

```bash
pip install -r requirements.txt
jupyter notebook project.ipynb
```

The first cell in the notebook also auto-installs dependencies via `subprocess`.

## Environment

- Python 3.10+; `random.seed(7)` and `np.random.seed(7)` for reproducibility
- Dataset: `keremberke/aerial-airport-object-detection` loaded via HuggingFace `datasets`
- YOLO model: `yolov8n.pt` (auto-downloaded by ultralytics on first run)
- Device: auto-detected (`cuda` if available, else `cpu`)
- Fine-tuning workspace written to `ft_workspace/` (created at runtime, gitignore-able)

## Key Design Decisions

- **GT labels**: dataset provides COCO `[x,y,w,h]` bboxes → converted to `[x1,y1,x2,y2]` via `get_gt_boxes_xyxy()`
- **YOLO recall**: computed with IoU threshold 0.5 against dataset GT boxes (not YOLO's internal mAP)
- **ORB metric**: raw keypoint count (Part 1 baseline) and matching ratio vs clean (SNR sweep)
- **Edge metric**: `edges.sum() / edges.size` — relative density ratio vs clean baseline
- **Fine-tune labels**: pseudo-labels from clean pretrained YOLO predictions (conf≥0.35), not dataset GT
- **Fine-tune distortion**: GaussNoise only (easiest to generate consistent training labels for)

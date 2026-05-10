# ProTracker: Prompt-Optimized Multi-Vessel Tracking for UAV Videos

ProTracker is a prompt-oriented multi-object vessel tracking framework designed for UAV-based waterway scenes. It integrates vessel detection, graph-based prompt optimization, target-aware refinement, and SAM2-based video segmentation to achieve robust and fine-grained multi-vessel tracking under challenging conditions such as dense vessel distribution, small targets, occlusion, water reflection, and complex illumination.

## Overview

Existing multi-object tracking methods usually rely on detection boxes and handcrafted association strategies. However, in UAV waterway scenarios, vessels are often small, densely distributed, visually similar, and frequently occluded. These factors can lead to missed detections, inaccurate bounding boxes, identity switches, and fragmented trajectories.

To address these problems, ProTracker introduces a prompt-optimized tracking pipeline. The framework first detects vessel candidates using a YOLO-based detector, then constructs a graph structure to model temporal and spatial relationships between vessel candidates. A graph-based cascaded prompt optimizer is used to enhance node representations and dynamically refine edge features. The optimized prompts are then used to guide SAM2 for fine-grained vessel segmentation and tracking.

## Framework

The overall pipeline of ProTracker contains four main components:

1. **Vessel Detection Module**  
   A YOLO11-based detector is used to obtain initial vessel candidates from UAV video frames.

2. **Graph-Based Cascaded Prompt Optimizer**  
   The detected vessel candidates and tracklets are modeled as graph nodes, while their temporal, spatial, motion, and appearance relationships are modeled as graph edges. The optimizer enhances node representations and refines edge features through iterative graph message passing.

3. **Target-Aware Refinement Module**  
   This module refines low-quality vessel prompts and supplements missing or weak vessel instances, especially for small targets and partially occluded vessels.

4. **SAM2-Based Tracking Module**  
   The optimized prompts are fed into SAM2 to generate fine-grained vessel masks and maintain temporal consistency across video frames.

## Project Structure

```text
ProTracker/
│
├── README.md
├── requirements.txt
├── environment.yml
├── run.py
├── train.py
├── test.py
│
├── assets/
│   ├── framework.png
│   ├── vesselmot_examples.png
│   └── qualitative_results.png
│
├── configs/
│   ├── train/
│   │   └── ProTracker.yaml
│   └── test/
│       └── ProTracker.yaml
│
├── datasets/
│   ├── README.md
│   └── Vessel-MOT/
│
├── weights/
│   ├── README.md
│   ├── yolo11/
│   ├── sam2/
│   └── protracker/
│
├── models/
│   ├── protracker.py
│   ├── yolo_detector.py
│   ├── sam2_predictor.py
│   └── target_refinement.py
│
├── networks/
│   ├── motmpnet.py
│   ├── graph_transformer.py
│   ├── edge_attention.py
│   ├── message_passing.py
│   ├── time_aware_node_model.py
│   └── prompt_optimizer.py
│
├── tracker/
│   ├── inference_pipeline.py
│   ├── track_manager.py
│   ├── association.py
│   └── prompt_generator.py
│
├── scripts/
│   ├── prepare_vesselmot.py
│   ├── convert_yolo_to_mot.py
│   ├── visualize_results.py
│   └── demo_video.py
│
├── evaluate/
│   ├── eval_mota.py
│   ├── eval_hota.py
│   └── eval_idf1.py
│
├── utils/
│   ├── video_io.py
│   ├── box_ops.py
│   ├── mask_ops.py
│   ├── mot_format.py
│   └── logger.py
│
└── external/
    ├── ultralytics/
    ├── SUSHI/
    └── sam2/
```

## Installation

This project is developed with Python and PyTorch. The recommended environment is:

- Python 3.10
- PyTorch 2.x
- CUDA 11.x or 12.x
- torchvision
- torch-geometric
- OpenCV
- NumPy

Install the required packages:

```bash
pip install -r requirements.txt
```

The third-party repositories should be placed under the `external/` directory:

```text
external/ultralytics/
external/SUSHI/
external/sam2/
```

## Dataset Preparation

The Vessel-MOT dataset should be organized as follows:

```text
datasets/Vessel-MOT/
├── images/
├── labels/
├── videos/
└── annotations/
```

The dataset contains UAV-based waterway videos with dense vessel targets, small objects, occlusion, water reflection, illumination changes, and complex backgrounds.

## Training

To train the graph-based cascaded prompt optimizer:

```bash
python train.py --config configs/train/ProTracker.yaml
```

The training configuration can be modified in:

```text
configs/train/ProTracker.yaml
```

## Testing

To test ProTracker on Vessel-MOT:

```bash
python test.py --config configs/test/ProTracker.yaml --weights weights/protracker/protracker.pth
```

The tracking results will be saved in:

```text
results/
```

## Demo

To run ProTracker on a single video:

```bash
python run.py --video assets/demo.mp4 --output results/demo
```

## Evaluation

The tracking performance can be evaluated using common multi-object tracking metrics, including:

- MOTA
- HOTA
- IDF1
- DetA
- AssA
- IDSW
- MT
- ML

Example command:

```bash
python evaluate/eval_mota.py --gt datasets/Vessel-MOT/annotations --pred results/
```

## Results

Qualitative tracking results can be visualized by:

```bash
python scripts/visualize_results.py --video assets/demo.mp4 --result results/demo.txt --output results/demo_vis.mp4
```

Example results will be shown in the `assets/` directory.

## Acknowledgements

This project builds upon several open-source projects, including Ultralytics YOLO, SUSHI, and SAM2. The original third-party repositories are placed under the `external/` directory. The main implementation of ProTracker is located in the `models/`, `networks/`, and `tracker/` directories, including the graph-based cascaded prompt optimizer, target-aware refinement module, prompt generation pipeline, and SAM2-based tracking pipeline.

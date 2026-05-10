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

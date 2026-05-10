# Project Structure

This document explains the directory structure of ProTracker.

ProTracker is a prompt-oriented multi-vessel tracking framework for UAV-based waterway scenes. It integrates YOLO11-based vessel detection, SUSHI-style graph-based association, and SAM2-based video segmentation into a unified multi-object tracking pipeline.

## Overall Structure

```text
ProTracker/
├── README.md
├── requirements.txt
├── run.py
├── train.py
├── test.py
├── LICENSE
├── THIRD_PARTY_NOTICES.md
│
├── configs/
│   ├── train/
│   │   └── ProTracker.yaml
│   └── test/
│       └── ProTracker.yaml
│
├── models/
│   ├── __init__.py
│   ├── protracker.py
│   ├── yolo_detector.py
│   ├── sam2_predictor.py
│   └── target_refinement.py
│
├── networks/
│   ├── __init__.py
│   ├── graph_transformer.py
│   ├── edge_attention.py
│   ├── message_passing.py
│   ├── time_aware_node_model.py
│   ├── motmpnet.py
│   └── prompt_optimizer.py
│
├── tracker/
│   ├── __init__.py
│   ├── inference_pipeline.py
│   ├── track_manager.py
│   ├── association.py
│   └── prompt_generator.py
│
├── utils/
│   ├── __init__.py
│   ├── video_io.py
│   ├── box_ops.py
│   ├── mask_ops.py
│   ├── mot_format.py
│   └── logger.py
│
├── scripts/
│   ├── prepare_vesselmot.py
│   ├── convert_yolo_to_mot.py
│   ├── visualize_results.py
│   └── demo_video.py
│
├── evaluate/
│   ├── README.md
│   ├── eval_mota.py
│   ├── eval_idf1.py
│   └── eval_hota.py
│
├── datasets/
│   └── README.md
│
├── weights/
│   └── README.md
│
├── assets/
│   ├── README.md
│   ├── framework.png
│   ├── vesselmot_challenging_scenes.png
│   ├── protracker_refinement_modules.png
│   └── protracker_qualitative_comparison.png
│
├── docs/
│   ├── README.md
│   ├── project_structure.md
│   ├── installation.md
│   ├── dataset_preparation.md
│   ├── training.md
│   ├── testing.md
│   └── evaluation.md
│
└── external/
    ├── ultralytics/
    ├── SUSHI/
    └── sam2/
```

## Root Files

### README.md

The main project introduction file.

It introduces the motivation, overall framework, installation process, dataset preparation, training, testing, and evaluation commands.

### requirements.txt

The dependency list of ProTracker.

It includes PyTorch, OpenCV, NumPy, PyYAML, Ultralytics, PyTorch Geometric related packages, and other required libraries.

### run.py

The unified demo inference entrance.

It is used to run ProTracker on a single UAV video:

```bash
python run.py --video assets/demo.mp4 --output results/demo
```

### train.py

The training entrance.

It is mainly used to train the graph-based cascaded prompt optimizer:

```bash
python train.py --config configs/train/ProTracker.yaml
```

### test.py

The testing entrance.

It is used to run ProTracker on the Vessel-MOT testing set:

```bash
python test.py --config configs/test/ProTracker.yaml
```

## configs

The `configs/` directory stores training and testing configuration files.

```text
configs/
├── train/
│   └── ProTracker.yaml
└── test/
    └── ProTracker.yaml
```

The training configuration includes dataset paths, optimizer settings, learning rate, batch size, and graph network settings.

The testing configuration includes video paths, model weight paths, tracking parameters, visualization settings, and evaluation settings.

## models

The `models/` directory contains high-level model wrappers.

```text
models/
├── protracker.py
├── yolo_detector.py
├── sam2_predictor.py
└── target_refinement.py
```

### yolo_detector.py

This file wraps the YOLO11 detector.

It converts raw YOLO outputs into a unified detection format used by ProTracker.

### sam2_predictor.py

This file wraps the SAM2 video predictor.

It receives optimized prompts and outputs vessel masks and video segmentation results.

### target_refinement.py

This file implements the target-aware refinement module.

It refines small-target prompts, occluded-target prompts, and missing-target prompts before sending them to SAM2.

### protracker.py

This file defines the main ProTracker model.

It connects:

```text
YOLO11 detector
→ graph-based cascaded prompt optimizer
→ target-aware refinement module
→ SAM2 predictor
```

## networks

The `networks/` directory contains graph neural network modules.

```text
networks/
├── graph_transformer.py
├── edge_attention.py
├── message_passing.py
├── time_aware_node_model.py
├── motmpnet.py
└── prompt_optimizer.py
```

### graph_transformer.py

This file implements the graph Transformer encoder layer.

It enhances node representations through graph self-attention.

### edge_attention.py

This file implements the dynamic edge attention module.

It computes edge weights according to the features of the two connected nodes.

### message_passing.py

This file contains the basic message passing components, including MLP, MetaLayer, EdgeModel, and MLPGraphIndependent.

### time_aware_node_model.py

This file implements time-aware node update.

It separates forward and backward temporal flows to update node features.

### motmpnet.py

This file implements the main message passing network.

It performs node encoding, edge encoding, iterative graph message passing, and edge classification.

### prompt_optimizer.py

This file wraps the graph neural network into the graph-based cascaded prompt optimizer.

It converts detection results into graph data and generates optimized vessel prompts for SAM2.

## tracker

The `tracker/` directory contains the tracking pipeline and track management modules.

```text
tracker/
├── inference_pipeline.py
├── track_manager.py
├── association.py
└── prompt_generator.py
```

### inference_pipeline.py

This file defines the complete inference pipeline.

It connects video loading, detection, prompt optimization, target refinement, SAM2 tracking, result saving, and visualization.

### track_manager.py

This file manages vessel tracks.

It creates new tracks, updates existing tracks, stores historical trajectories, and handles lost tracks.

### association.py

This file provides the association module.

It matches detections and tracks using IoU, center distance, confidence, and optional graph edge scores.

### prompt_generator.py

This file converts detections, tracks, and graph outputs into SAM2-compatible prompts.

## utils

The `utils/` directory contains common utility functions.

```text
utils/
├── video_io.py
├── box_ops.py
├── mask_ops.py
├── mot_format.py
└── logger.py
```

These files support video processing, bounding box operations, mask processing, MOT-format conversion, and logging.

## scripts

The `scripts/` directory contains useful command-line tools.

```text
scripts/
├── prepare_vesselmot.py
├── convert_yolo_to_mot.py
├── visualize_results.py
└── demo_video.py
```

### prepare_vesselmot.py

Creates and checks the Vessel-MOT dataset directory structure.

### convert_yolo_to_mot.py

Converts YOLO-format annotations to MOT-format annotations.

### visualize_results.py

Draws tracking results on videos or image sequences.

### demo_video.py

Runs a quick demo by calling `run.py`.

## evaluate

The `evaluate/` directory contains lightweight evaluation scripts.

```text
evaluate/
├── eval_mota.py
├── eval_idf1.py
└── eval_hota.py
```

These scripts calculate common MOT metrics, including MOTA, IDF1, HOTA, DetA, AssA, FP, FN, and IDSW.

For strict benchmark evaluation, the official TrackEval toolkit is recommended.

## datasets

The `datasets/` directory is used to store the Vessel-MOT dataset.

The complete dataset is not included in this repository because of storage limitations.

Expected structure:

```text
datasets/
└── Vessel-MOT/
    ├── images/
    ├── labels/
    ├── videos/
    └── annotations/
```

## weights

The `weights/` directory stores pretrained weights and trained checkpoints.

Expected structure:

```text
weights/
├── yolo11/
│   └── yolo11_vessel.pt
├── sam2/
│   └── sam2_hiera_large.pt
└── protracker/
    └── protracker.pth
```

Large weight files are not included in the repository.

## assets

The `assets/` directory stores images and demo materials used in the README.

Recommended files:

```text
assets/
├── framework.png
├── vesselmot_challenging_scenes.png
├── protracker_refinement_modules.png
└── protracker_qualitative_comparison.png
```

## external

The `external/` directory stores third-party source code.

```text
external/
├── ultralytics/
├── SUSHI/
└── sam2/
```

These projects are used as third-party dependencies or references.

Please refer to `THIRD_PARTY_NOTICES.md` for license and source information.

## Unified Pipeline

Although ProTracker integrates three different projects, the final pipeline is organized as one unified framework:

```text
Input UAV video
    ↓
YOLO11 vessel detection
    ↓
Graph-based cascaded prompt optimization
    ↓
Target-aware prompt refinement
    ↓
SAM2 video segmentation and tracking
    ↓
MOT-format result output and visualization
```

This structure makes the project easier to understand, run, maintain, and extend.
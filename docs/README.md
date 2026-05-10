# Documentation

This directory contains additional documents for ProTracker.

ProTracker is a prompt-oriented multi-vessel tracking framework for UAV-based waterway scenes. The main repository already provides the code, configuration files, and running scripts. This directory is used to provide more detailed explanations about the project structure, dataset preparation, training, testing, and evaluation.

## Directory Structure

```text
docs/
├── README.md
├── project_structure.md
├── installation.md
├── dataset_preparation.md
├── training.md
├── testing.md
└── evaluation.md
```

## Documents

### project_structure.md

This document explains the overall file organization of ProTracker.

It describes how YOLO11, SUSHI-style graph modeling, and SAM2 are reorganized into a unified tracking framework.

### installation.md

This document introduces the environment setup process, including Python dependencies, PyTorch, PyTorch Geometric, Ultralytics YOLO, SAM2, and other required packages.

### dataset_preparation.md

This document explains how to prepare the Vessel-MOT dataset and how to organize images, videos, labels, and annotations.

### training.md

This document introduces how to train the graph-based cascaded prompt optimizer.

### testing.md

This document introduces how to run ProTracker on testing videos or a single demo video.

### evaluation.md

This document introduces how to evaluate tracking results using MOTA, HOTA, IDF1, DetA, AssA, IDSW, MT, and ML.

## Notes

The documents in this directory are mainly used to make the project easier to understand and reproduce.

For a quick start, please refer to the main `README.md` in the root directory.
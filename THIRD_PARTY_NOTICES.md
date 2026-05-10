# Third-Party Notices

This project, **ProTracker**, integrates and adapts components from several open-source projects.

ProTracker is developed for high-density vessel tracking in UAV-based waterway scenarios. The project combines object detection, graph-based data association, prompt optimization, and video segmentation into a unified tracking framework.

The following third-party projects are used or referenced in this repository.

---

## 1. Ultralytics YOLO

**Project:** Ultralytics YOLO  
**Repository:** https://github.com/ultralytics/ultralytics  
**License:** AGPL-3.0 License

Ultralytics YOLO is used as the vessel detection module in ProTracker. In this project, YOLO is responsible for detecting vessel candidates in UAV video frames and providing initial bounding boxes for the following graph-based prompt optimization module.

In ProTracker, the YOLO-related code and model interface are mainly organized under:

```text
external/ultralytics/
models/yolo_detector.py
```

The pretrained or fine-tuned YOLO weights should be placed under:

```text
weights/yolo11/
```

Please refer to the original Ultralytics repository and license for detailed usage and license requirements.

---

## 2. SUSHI

**Project:** SUSHI: Unifying Short and Long-Term Tracking with Graph Hierarchies  
**Repository:** https://github.com/dvl-tum/SUSHI  
**License:** MIT License

SUSHI is used as an important reference for hierarchical graph-based multi-object tracking. ProTracker refers to its graph-based association idea and adapts the message-passing structure for UAV-based multi-vessel tracking.

In ProTracker, the SUSHI-related graph modeling idea is reorganized into the graph-based cascaded prompt optimizer, mainly including:

```text
networks/motmpnet.py
networks/message_passing.py
networks/time_aware_node_model.py
networks/prompt_optimizer.py
```

Compared with the original SUSHI framework, ProTracker focuses on vessel tracking in UAV waterway scenes and further combines graph-based association with SAM2 prompt generation.

---

## 3. Segment Anything Model 2

**Project:** Segment Anything Model 2  
**Repository:** https://github.com/facebookresearch/sam2  
**License:** Apache License 2.0

SAM2 is used as the video segmentation and tracking module in ProTracker. It provides fine-grained mask prediction and video object propagation ability.

In ProTracker, SAM2 is mainly used after prompt optimization. The graph-based prompt optimizer generates high-quality vessel prompts, and SAM2 uses these prompts to produce more accurate vessel masks and tracking results.

The SAM2-related code and model interface are mainly organized under:

```text
external/sam2/
models/sam2_predictor.py
```

The SAM2 checkpoints should be placed under:

```text
weights/sam2/
```

Please refer to the original SAM2 repository and license for detailed usage and license requirements.

---

## 4. PyTorch Geometric and Torch Scatter

**Projects:**

- PyTorch Geometric
- torch-scatter

These libraries are used for graph neural network implementation, including graph convolution, message passing, and feature aggregation.

They are mainly used in:

```text
networks/graph_transformer.py
networks/message_passing.py
networks/time_aware_node_model.py
networks/motmpnet.py
```

Please refer to their official repositories and licenses for details.

---

## 5. OpenCV

**Project:** OpenCV  
**Repository:** https://github.com/opencv/opencv  
**License:** Apache License 2.0

OpenCV is used for video reading, frame processing, visualization, and result saving.

It is mainly used in:

```text
utils/video_io.py
scripts/visualize_results.py
tracker/inference_pipeline.py
```

---

## 6. NumPy

**Project:** NumPy  
**Repository:** https://github.com/numpy/numpy  
**License:** BSD License

NumPy is used for numerical calculation, bounding box processing, mask processing, and result conversion.

It is mainly used in:

```text
utils/box_ops.py
utils/mask_ops.py
utils/mot_format.py
tracker/association.py
tracker/track_manager.py
```

---

## Notes

This repository is intended for academic research and graduation project demonstration.

If you use this project, please also comply with the licenses of all third-party components listed above.

Large pretrained weights, datasets, and experimental results are not included in this repository. Please download or prepare them separately and place them under the corresponding directories.
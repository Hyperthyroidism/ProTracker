# Testing

This document explains how to test ProTracker on UAV vessel videos.

ProTracker supports two testing modes:

1. Testing on a single demo video
2. Testing on the Vessel-MOT test set

## 1. Testing Entrance

The testing entrance is:

```text
test.py
```

Run testing with:

```bash
python test.py --config configs/test/ProTracker.yaml
```

You can also specify the device:

```bash
python test.py \
  --config configs/test/ProTracker.yaml \
  --device cuda
```

If CUDA is not available:

```bash
python test.py \
  --config configs/test/ProTracker.yaml \
  --device cpu
```

## 2. Single Video Inference

For a single video, use:

```text
run.py
```

Example:

```bash
python run.py \
  --config configs/test/ProTracker.yaml \
  --video assets/demo.mp4 \
  --output results/demo
```

If you want to run on CPU:

```bash
python run.py \
  --config configs/test/ProTracker.yaml \
  --video assets/demo.mp4 \
  --output results/demo \
  --device cpu
```

## 3. Demo Script

A quick demo script is also provided:

```text
scripts/demo_video.py
```

Run:

```bash
python scripts/demo_video.py \
  --video assets/demo.mp4 \
  --output results/demo
```

This script internally calls `run.py`.

## 4. Testing Configuration

The testing configuration file is:

```text
configs/test/ProTracker.yaml
```

The main testing settings include:

```yaml
detector:
  enabled: true
  model_path: weights/yolo11/yolo11_vessel.pt
  image_size: 640
  confidence_threshold: 0.25
  nms_threshold: 0.70

graph_prompt_optimizer:
  enabled: true
  checkpoint_path: weights/protracker/protracker.pth

sam2:
  enabled: true
  config_path: external/sam2/configs/sam2_hiera_l.yaml
  checkpoint_path: weights/sam2/sam2_hiera_large.pt
  prompt_type: box

tracking:
  result_dir: results/test
  association_threshold: 0.5
  max_lost_frames: 30
  min_track_length: 3

visualization:
  enabled: true
  show_box: true
  show_mask: true
  show_track_id: true
```

If your weight files are stored in different locations, modify the corresponding paths in the configuration file.

## 5. Required Weights

Before testing, prepare the following weights:

```text
weights/
├── yolo11/
│   └── yolo11_vessel.pt
├── sam2/
│   └── sam2_hiera_large.pt
└── protracker/
    └── protracker.pth
```

### YOLO11 Weight

The YOLO11 weight is used for vessel detection.

```text
weights/yolo11/yolo11_vessel.pt
```

### SAM2 Weight

The SAM2 weight is used for video segmentation and mask propagation.

```text
weights/sam2/sam2_hiera_large.pt
```

### ProTracker Weight

The ProTracker weight is used for the graph-based cascaded prompt optimizer.

```text
weights/protracker/protracker.pth
```

If the graph prompt optimizer checkpoint is not available, the framework can still run with detection-based prompts, but the tracking performance may be reduced.

## 6. Testing on Vessel-MOT

The test videos should be placed under:

```text
datasets/Vessel-MOT/videos/test/
```

Example:

```text
datasets/Vessel-MOT/videos/test/
├── sequence_001.mp4
├── sequence_002.mp4
└── sequence_003.mp4
```

Run:

```bash
python test.py --config configs/test/ProTracker.yaml
```

The results will be saved under:

```text
results/test/
```

Example:

```text
results/test/
├── sequence_001.txt
├── sequence_001_vis.mp4
├── sequence_002.txt
├── sequence_002_vis.mp4
└── metrics.json
```

## 7. Output Format

The tracking results are saved in MOT format:

```text
frame_id, track_id, x, y, width, height, confidence, class_id, visibility
```

Example:

```text
1,1,512.400000,236.700000,42.500000,18.300000,1.000000,0,1.000000
```

Field description:

```text
frame_id: frame index, usually starting from 1
track_id: vessel identity number
x: top-left x coordinate
y: top-left y coordinate
width: bounding box width
height: bounding box height
confidence: tracking confidence
class_id: object category id
visibility: visibility ratio
```

For this project, the class id is usually:

```text
0: vessel
```

## 8. Visualization Output

If visualization is enabled, ProTracker will generate visualized videos.

Example:

```text
results/demo/demo_vis.mp4
```

The visualized video may include:

```text
tracking bounding boxes
track identity numbers
confidence scores
segmentation masks
```

Visualization settings can be modified in:

```text
configs/test/ProTracker.yaml
```

Example:

```yaml
visualization:
  enabled: true
  show_box: true
  show_mask: true
  show_track_id: true
  show_confidence: false
  line_width: 2
```

## 9. Visualize Existing Results

If you already have a MOT-format result file, you can visualize it by running:

```bash
python scripts/visualize_results.py \
  --video assets/demo.mp4 \
  --result results/demo/demo.txt \
  --output results/demo/demo_vis.mp4
```

For image sequences:

```bash
python scripts/visualize_results.py \
  --image-dir datasets/Vessel-MOT/images/test/sequence_001 \
  --result results/test/sequence_001.txt \
  --output results/test/sequence_001_vis
```

## 10. Run Evaluation After Testing

If evaluation is enabled in the config file:

```yaml
evaluation:
  enabled: true
  gt_dir: datasets/Vessel-MOT/annotations/test
  pred_dir: results/test
```

Then `test.py` will evaluate the prediction results after testing.

The evaluation results may include:

```text
MOTA
HOTA
IDF1
DetA
AssA
IDSW
MT
ML
FP
FN
```

## 11. Common Problems

### Weight file not found

Check whether the weight paths in the config file are correct.

Example:

```yaml
detector:
  model_path: weights/yolo11/yolo11_vessel.pt
```

### SAM2 config not found

Check this path:

```yaml
sam2:
  config_path: external/sam2/configs/sam2_hiera_l.yaml
```

If your SAM2 config file has a different name, modify the config path.

### No tracking result generated

Possible reasons:

```text
1. YOLO confidence threshold is too high.
2. YOLO weight is not correctly loaded.
3. Input video path is wrong.
4. SAM2 prompt initialization failed.
5. No vessel is detected in the initial frames.
```

You can try lowering the confidence threshold:

```yaml
detector:
  confidence_threshold: 0.15
```

### Visualization video cannot be opened

Make sure OpenCV supports the selected video codec.

The default codec is:

```text
mp4v
```

If it does not work on your system, try changing the output format or codec in the visualization script.

## 12. Recommended Testing Command

For a quick test:

```bash
python run.py \
  --config configs/test/ProTracker.yaml \
  --video assets/demo.mp4 \
  --output results/demo
```

For full test set evaluation:

```bash
python test.py \
  --config configs/test/ProTracker.yaml
```
# Evaluation

This document explains how to evaluate ProTracker tracking results.

ProTracker provides lightweight evaluation scripts for common multi-object tracking metrics, including:

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

The evaluation scripts are located in:

```text
evaluate/
├── README.md
├── eval_mota.py
├── eval_idf1.py
└── eval_hota.py
```

## 1. Evaluation Input Format

Both ground truth files and prediction files should follow the MOT format:

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
track_id: object identity number
x: top-left x coordinate of the bounding box
y: top-left y coordinate of the bounding box
width: bounding box width
height: bounding box height
confidence: detection or tracking confidence
class_id: object category id
visibility: visibility ratio
```

For Vessel-MOT, the category is usually:

```text
0: vessel
```

## 2. Directory Structure

The ground truth annotations should be placed under:

```text
datasets/Vessel-MOT/annotations/test/
```

The prediction results should be placed under:

```text
results/test/
```

Example:

```text
datasets/Vessel-MOT/annotations/test/
├── sequence_001.txt
├── sequence_002.txt
└── sequence_003.txt

results/test/
├── sequence_001.txt
├── sequence_002.txt
└── sequence_003.txt
```

The file names of ground truth and prediction results should be the same.

For example:

```text
Ground truth: datasets/Vessel-MOT/annotations/test/sequence_001.txt
Prediction:   results/test/sequence_001.txt
```

## 3. Evaluate MOTA

MOTA is used to measure the overall tracking accuracy. It mainly considers false positives, false negatives, and identity switches.

Run:

```bash
python evaluate/eval_mota.py \
  --gt datasets/Vessel-MOT/annotations/test \
  --pred results/test \
  --iou-threshold 0.5
```

The script reports:

```text
MOTA
GT
Matches
FP
FN
IDSW
```

## 4. Evaluate IDF1

IDF1 is used to measure identity consistency. It reflects whether the tracker can keep the same identity for the same vessel across frames.

Run:

```bash
python evaluate/eval_idf1.py \
  --gt datasets/Vessel-MOT/annotations/test \
  --pred results/test \
  --iou-threshold 0.5
```

The script reports:

```text
IDF1
IDP
IDR
IDTP
IDFP
IDFN
```

## 5. Evaluate HOTA

HOTA evaluates both detection accuracy and association accuracy.

Run:

```bash
python evaluate/eval_hota.py \
  --gt datasets/Vessel-MOT/annotations/test \
  --pred results/test \
  --iou-threshold 0.5
```

The script reports:

```text
HOTA
DetA
AssA
TP
FP
FN
IDTP
IDFP
IDFN
```

## 6. Run Evaluation in test.py

The testing script can automatically run evaluation after inference.

In the testing configuration file:

```text
configs/test/ProTracker.yaml
```

set:

```yaml
evaluation:
  enabled: true
  gt_dir: datasets/Vessel-MOT/annotations/test
  pred_dir: results/test
```

Then run:

```bash
python test.py --config configs/test/ProTracker.yaml
```

After testing, the script will evaluate the tracking results automatically.

## 7. Evaluation Metrics

### MOTA

MOTA measures the overall tracking accuracy.

It considers:

```text
false positives
false negatives
identity switches
```

A higher MOTA value means better overall tracking performance.

### IDF1

IDF1 measures identity preservation ability.

A higher IDF1 means that the tracker can better keep the same identity for the same object across video frames.

### HOTA

HOTA balances detection accuracy and association accuracy.

It can be regarded as a more balanced metric for multi-object tracking.

### DetA

DetA measures detection accuracy.

It reflects whether objects are correctly detected and localized.

### AssA

AssA measures association accuracy.

It reflects whether detections belonging to the same object are correctly linked into the same trajectory.

### IDSW

IDSW means identity switches.

A lower IDSW value means fewer identity changes.

### MT

MT means mostly tracked targets.

It counts the number of ground-truth trajectories that are successfully tracked for most of their lifespan.

### ML

ML means mostly lost targets.

It counts the number of ground-truth trajectories that are lost for most of their lifespan.

## 8. Notes on Lightweight Evaluation

The evaluation scripts in this repository are lightweight implementations.

They are mainly used for:

```text
project demonstration
quick result checking
basic metric calculation
GitHub repository completeness
```

For strict benchmark-level evaluation, it is recommended to use the official TrackEval toolkit.

The prediction files generated by ProTracker follow the MOT format and can be further adapted to TrackEval if needed.

## 9. Common Problems

### No matched sequence files found

Check whether the file names in ground truth and prediction directories are the same.

Example:

```text
datasets/Vessel-MOT/annotations/test/sequence_001.txt
results/test/sequence_001.txt
```

### Prediction file missing

If the evaluation script prints:

```text
Prediction file missing for sequence: sequence_001.txt
```

check whether the prediction result exists under:

```text
results/test/
```

### MOTA is negative

MOTA can be negative when false positives, false negatives, and identity switches are large.

This usually means that the prediction results are poor or the prediction format is incorrect.

### IDF1 is very low

Possible reasons:

```text
1. Track IDs are unstable.
2. The same vessel is assigned different IDs in different frames.
3. Prediction boxes do not match ground truth boxes.
4. Ground truth and prediction frame IDs are not aligned.
```

### HOTA is very low

Possible reasons:

```text
1. Detection accuracy is low.
2. Association accuracy is low.
3. IoU threshold is too strict.
4. Result format is incorrect.
```

## 10. Recommended Evaluation Commands

Evaluate MOTA:

```bash
python evaluate/eval_mota.py \
  --gt datasets/Vessel-MOT/annotations/test \
  --pred results/test
```

Evaluate IDF1:

```bash
python evaluate/eval_idf1.py \
  --gt datasets/Vessel-MOT/annotations/test \
  --pred results/test
```

Evaluate HOTA:

```bash
python evaluate/eval_hota.py \
  --gt datasets/Vessel-MOT/annotations/test \
  --pred results/test
```
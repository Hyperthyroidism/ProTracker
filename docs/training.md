# Training

This document explains how to train the graph-based cascaded prompt optimizer in ProTracker.

In ProTracker, YOLO11 and SAM2 are mainly used as pretrained modules. The main trainable component is the graph-based cascaded prompt optimizer, which learns to model vessel association relationships and generate stable prompts for SAM2.

## 1. Training Target

The training process mainly focuses on:

```text
Graph-based Cascaded Prompt Optimizer
```

This module is implemented in:

```text
networks/prompt_optimizer.py
networks/motmpnet.py
networks/message_passing.py
networks/time_aware_node_model.py
networks/edge_attention.py
networks/graph_transformer.py
```

The optimizer receives vessel detections or tracklets as graph nodes, constructs edges between candidate associations, and predicts whether each edge represents a correct tracking association.

## 2. Training Entrance

The training entrance is:

```text
train.py
```

Run training by:

```bash
python train.py --config configs/train/ProTracker.yaml
```

You can also specify the device:

```bash
python train.py \
  --config configs/train/ProTracker.yaml \
  --device cuda
```

If CUDA is not available:

```bash
python train.py \
  --config configs/train/ProTracker.yaml \
  --device cpu
```

## 3. Training Configuration

The training configuration file is:

```text
configs/train/ProTracker.yaml
```

The main training settings include:

```yaml
training:
  epochs: 200
  batch_size: 4
  num_workers: 4
  save_interval: 10
  log_interval: 20
```

Optimizer settings:

```yaml
optimizer:
  type: Adam
  learning_rate: 0.0003
  weight_decay: 0.0001
```

Graph prompt optimizer settings:

```yaml
graph_prompt_optimizer:
  enabled: true
  checkpoint_path: weights/protracker/protracker.pth

  graph:
    node_feature_dim: 256
    edge_feature_dim: 32
    hidden_dim: 128
    max_temporal_distance: 30
    max_neighbors: 10

  message_passing:
    num_steps: 6
    dropout: 0.1
    node_agg_fn: mean

  edge_attention:
    enabled: true
    activation: sigmoid
```

## 4. Dataset Requirement

The training data should be placed under:

```text
datasets/Vessel-MOT/
```

Recommended structure:

```text
datasets/Vessel-MOT/
├── images/
│   └── train/
├── labels/
│   └── train/
└── annotations/
    └── train/
```

The training process needs:

```text
1. vessel detection boxes
2. frame indexes
3. object identity labels
4. temporal association relationships
```

These data are used to construct graph nodes, graph edges, and edge labels.

## 5. Graph Construction

During training, each detection or tracklet is treated as a graph node.

A graph node usually contains:

```text
bounding box position
detection confidence
appearance feature
frame index
track identity
```

Candidate graph edges are built between nodes that may belong to the same vessel identity.

An edge usually contains:

```text
relative position
center distance
box size difference
IoU
temporal distance
appearance similarity
motion consistency
```

The graph neural network predicts whether each edge is a correct association.

## 6. Loss Function

The graph optimizer is trained as an edge classification model.

For each edge, the label is:

```text
1: the two nodes belong to the same vessel identity
0: the two nodes belong to different vessel identities
```

The default loss is binary cross-entropy loss:

```text
BCEWithLogitsLoss
```

The loss is implemented in:

```text
networks/prompt_optimizer.py
```

Function:

```python
compute_loss()
```

## 7. Output Checkpoints

Training checkpoints are saved under:

```text
weights/protracker/
```

Example:

```text
weights/protracker/protracker_epoch_010.pth
weights/protracker/protracker_epoch_020.pth
weights/protracker/protracker_best.pth
```

The final checkpoint can be renamed as:

```text
weights/protracker/protracker.pth
```

The testing configuration should point to this checkpoint:

```yaml
graph_prompt_optimizer:
  checkpoint_path: weights/protracker/protracker.pth
```

## 8. Logs

Training logs are saved under:

```text
logs/train/
```

Example:

```text
logs/train/train.log
```

The log records:

```text
training epoch
training loss
edge classification loss
learning rate
checkpoint path
```

## 9. Resume Training

If resume training is supported, specify the checkpoint path in the configuration file:

```yaml
training:
  resume: true
  resume_path: weights/protracker/protracker_epoch_100.pth
```

Then run:

```bash
python train.py --config configs/train/ProTracker.yaml
```

## 10. Notes

The current training script mainly provides a clean project-level training interface.

If the full Vessel-MOT training annotation loader is not implemented, the graph optimizer can still be connected later by completing the dataset class and graph construction logic.

The core idea is:

```text
Vessel-MOT annotations
→ graph nodes and edges
→ MOTMPNet message passing
→ edge classification
→ optimized vessel prompts
```

## 11. Recommended Training Setting

The recommended training setting used in this project is:

```text
Framework: PyTorch
Epochs: 200
Optimizer: Adam
Learning rate: 0.0003
Weight decay: 0.0001
Batch size: 4
Image size for detector: 640
```

Other settings can follow the default values in the configuration file.
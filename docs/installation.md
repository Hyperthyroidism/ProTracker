# Installation

This document explains how to set up the environment for ProTracker.

ProTracker integrates YOLO11, graph-based message passing networks, and SAM2. Therefore, the environment should include PyTorch, OpenCV, Ultralytics, PyTorch Geometric, torch-scatter, and SAM2-related dependencies.

## 1. Create Conda Environment

It is recommended to use a separate conda environment.

```bash
conda create -n protracker python=3.10 -y
conda activate protracker
```

## 2. Install PyTorch

Install PyTorch according to your CUDA version.

For example, if CUDA 11.8 is used:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If CUDA is not available, install the CPU version:

```bash
pip install torch torchvision torchaudio
```

After installation, check whether PyTorch and CUDA are available:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 3. Install Basic Dependencies

Install common dependencies:

```bash
pip install -r requirements.txt
```

The main packages include:

```text
numpy
opencv-python
PyYAML
tqdm
matplotlib
scipy
pandas
ultralytics
```

## 4. Install PyTorch Geometric Dependencies

The graph-based cascaded prompt optimizer uses PyTorch Geometric and torch-scatter.

Install them according to your PyTorch and CUDA versions.

Example:

```bash
pip install torch-geometric
pip install torch-scatter
```

If installation fails, please install `torch-scatter` from the official wheel source that matches your PyTorch and CUDA version.

After installation, check the packages:

```bash
python -c "import torch_geometric; import torch_scatter; print('PyG installed successfully')"
```

## 5. Prepare Third-Party Repositories

The third-party repositories should be placed under:

```text
external/
├── ultralytics/
├── SUSHI/
└── sam2/
```

If the source code is already included in the repository, make sure the directory structure is correct.

If not, clone or copy the corresponding repositories into `external/`.

## 6. Install Ultralytics YOLO

If Ultralytics is installed through pip:

```bash
pip install ultralytics
```

If the source code is placed under `external/ultralytics/`, install it in editable mode:

```bash
pip install -e external/ultralytics
```

Check installation:

```bash
python -c "from ultralytics import YOLO; print('Ultralytics installed successfully')"
```

## 7. Install SAM2

If SAM2 source code is placed under:

```text
external/sam2/
```

install it in editable mode:

```bash
pip install -e external/sam2
```

Some SAM2 versions may require additional packages. If an error occurs, install the missing packages according to the error message.

Check installation:

```bash
python -c "import sam2; print('SAM2 installed successfully')"
```

## 8. Prepare Weights

The pretrained weights should be placed under:

```text
weights/
├── yolo11/
│   └── yolo11_vessel.pt
├── sam2/
│   └── sam2_hiera_large.pt
└── protracker/
    └── protracker.pth
```

The weight files are not included in this repository because they are usually large.

Please prepare them manually before running inference.

## 9. Prepare Dataset

The Vessel-MOT dataset should be placed under:

```text
datasets/Vessel-MOT/
```

Expected structure:

```text
datasets/Vessel-MOT/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── videos/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train/
    ├── val/
    └── test/
```

You can create and check the dataset directory by running:

```bash
python scripts/prepare_vesselmot.py --root datasets/Vessel-MOT --create --check --summary
```

## 10. Quick Test

After installation, run a demo video:

```bash
python run.py \
  --config configs/test/ProTracker.yaml \
  --video assets/demo.mp4 \
  --output results/demo
```

Or use the demo script:

```bash
python scripts/demo_video.py \
  --video assets/demo.mp4 \
  --output results/demo
```

## 11. Common Problems

### CUDA is not available

If CUDA is not available, the program will automatically use CPU. However, SAM2 and YOLO inference will be much slower.

You can also manually specify CPU:

```bash
python run.py --video assets/demo.mp4 --output results/demo --device cpu
```

### torch-scatter installation failed

`torch-scatter` must match the installed PyTorch and CUDA version.

If normal installation fails, install the corresponding wheel manually.

### SAM2 import failed

Make sure that SAM2 is installed correctly:

```bash
pip install -e external/sam2
```

Also make sure that the SAM2 checkpoint and config path in `configs/test/ProTracker.yaml` are correct.

### YOLO weight not found

Check whether the YOLO weight file exists:

```text
weights/yolo11/yolo11_vessel.pt
```

If your file name is different, modify the following field in the config file:

```yaml
detector:
  model_path: weights/yolo11/yolo11_vessel.pt
```

## 12. Recommended Environment

The recommended environment is:

```text
Python >= 3.10
PyTorch >= 2.0
CUDA >= 11.8
OpenCV >= 4.8
PyTorch Geometric
torch-scatter
Ultralytics
SAM2
```

The project can also run on CPU for basic code checking, but GPU is recommended for practical inference and testing.
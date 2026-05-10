\# External Dependencies



This directory stores third-party projects used or referenced by ProTracker.



ProTracker integrates YOLO11-based vessel detection, SUSHI-style graph-based association, and SAM2-based video segmentation into a unified multi-vessel tracking framework.



The source code in this directory is not the main original implementation of ProTracker. Instead, it is used as external support for detection, segmentation, and graph-based tracking references.



\## Directory Structure



```text

external/

├── README.md

├── ultralytics/

├── SUSHI/

└── sam2/

```



\## 1. Ultralytics YOLO



```text

external/ultralytics/

```



This directory stores the Ultralytics YOLO source code or related interface code.



In ProTracker, YOLO11 is used as the vessel detector. It detects vessel candidates from UAV video frames and provides initial bounding boxes for the following graph-based prompt optimization module.



The YOLO detector wrapper in ProTracker is located at:



```text

models/yolo\_detector.py

```



The YOLO weight file should be placed under:



```text

weights/yolo11/

```



Example:



```text

weights/yolo11/yolo11\_vessel.pt

```



\## 2. SUSHI



```text

external/SUSHI/

```



This directory stores the SUSHI source code or reference implementation.



SUSHI provides an important reference for hierarchical graph-based multi-object tracking. ProTracker adapts this idea into a graph-based cascaded prompt optimizer for UAV-based multi-vessel tracking.



The graph-related modules in ProTracker are reorganized under:



```text

networks/

├── graph\_transformer.py

├── edge\_attention.py

├── message\_passing.py

├── time\_aware\_node\_model.py

├── motmpnet.py

└── prompt\_optimizer.py

```



Compared with the original SUSHI framework, ProTracker focuses on vessel tracking in UAV waterway scenes and connects graph-based association with SAM2 prompt generation.



\## 3. SAM2



```text

external/sam2/

```



This directory stores the Segment Anything Model 2 source code or related interface code.



In ProTracker, SAM2 is used as the video segmentation and tracking module. It receives optimized vessel prompts and generates fine-grained vessel masks.



The SAM2 wrapper in ProTracker is located at:



```text

models/sam2\_predictor.py

```



The SAM2 checkpoint should be placed under:



```text

weights/sam2/

```



Example:



```text

weights/sam2/sam2\_hiera\_large.pt

```



\## Notes



The third-party projects in this directory may have their own licenses and usage requirements.



Please refer to the original repositories and the following file for more information:



```text

THIRD\_PARTY\_NOTICES.md

```



Large third-party source folders can also be replaced by installation instructions or Git submodules if necessary.



For example, if the repository becomes too large, only keep this README file and write the installation commands in:



```text

docs/installation.md

```


\# Dataset Preparation



This directory is used to store the Vessel-MOT dataset for ProTracker.



\## Dataset Structure



Please organize the dataset as follows:



```text

datasets/

└── Vessel-MOT/

&#x20;   ├── images/

&#x20;   │   ├── train/

&#x20;   │   ├── val/

&#x20;   │   └── test/

&#x20;   │

&#x20;   ├── labels/

&#x20;   │   ├── train/

&#x20;   │   ├── val/

&#x20;   │   └── test/

&#x20;   │

&#x20;   ├── videos/

&#x20;   │   ├── train/

&#x20;   │   ├── val/

&#x20;   │   └── test/

&#x20;   │

&#x20;   └── annotations/

&#x20;       ├── train/

&#x20;       ├── val/

&#x20;       └── test/

```



\## Description



Vessel-MOT is a UAV-based multi-vessel tracking dataset constructed for dense waterway scenarios. It contains challenging scenes such as:



\- Dense vessel distribution

\- Small vessel targets

\- Frequent occlusion

\- Water reflection

\- Complex illumination

\- Dynamic UAV viewpoints

\- Similar vessel appearances



The dataset is used to evaluate the robustness of ProTracker under complex real-world waterway conditions.



\## File Format



The tracking annotations follow the common MOT format:



```text

frame\_id, track\_id, x, y, width, height, confidence, class\_id, visibility

```



where:



```text

frame\_id      Frame index

track\_id      Identity number of the vessel

x             Top-left x coordinate of the bounding box

y             Top-left y coordinate of the bounding box

width         Width of the bounding box

height        Height of the bounding box

confidence    Detection or annotation confidence

class\_id      Object category, where 0 indicates vessel

visibility    Visibility ratio of the object

```



\## Example



A typical annotation line is:



```text

1, 3, 512.4, 236.7, 42.5, 18.3, 1.0, 0, 1.0

```



This means that in frame 1, vessel ID 3 is located at bounding box `(512.4, 236.7, 42.5, 18.3)`.



\## Usage



During training, the dataset path is specified in:



```text

configs/train/ProTracker.yaml

```



During testing and evaluation, the dataset path is specified in:



```text

configs/test/ProTracker.yaml

```



\## Notes



The complete Vessel-MOT dataset is not included in this repository due to storage limitations. Please place the dataset files in the corresponding folders before training or testing.


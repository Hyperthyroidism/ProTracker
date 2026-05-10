\# Assets



This directory stores figures, demo videos, and visualization materials used in the README and project documentation.



\## Directory Structure



```text

assets/

├── README.md

├── framework.png

├── vesselmot\_examples.png

├── qualitative\_results.png

└── demo.mp4

```



\## File Description



\### framework.png



```text

assets/framework.png

```



This figure shows the overall framework of ProTracker.



The framework contains four main components:



1\. YOLO11-based vessel detection module

2\. Graph-based cascaded prompt optimizer

3\. Target-aware refinement module

4\. SAM2-based vessel segmentation and tracking module



This image is mainly used in the README to help readers understand the complete pipeline of ProTracker.



\### vesselmot\_examples.png



```text

assets/vesselmot\_examples.png

```



This figure shows representative samples from the Vessel-MOT dataset.



The examples should include challenging UAV-based waterway scenes, such as:



\- Dense vessel distribution

\- Small vessel targets

\- Vessel occlusion

\- Water reflection

\- Complex illumination

\- Similar vessel appearances

\- Dynamic UAV viewpoints



\### qualitative\_results.png



```text

assets/qualitative\_results.png

```



This figure shows qualitative tracking results of ProTracker.



It is recommended to include tracking results with bounding boxes, masks, and identity numbers to demonstrate the effectiveness of the proposed method.



\### demo.mp4



```text

assets/demo.mp4

```



This is a short demo video used for quick inference testing.



The demo can be executed by:



```bash

python run.py --video assets/demo.mp4 --output results/demo

```



or:



```bash

python scripts/demo\_video.py --video assets/demo.mp4 --output results/demo

```



\## Notes



Large videos and high-resolution visualization files are not recommended to be uploaded directly to GitHub.



If the file size is too large, please compress the video or only upload several representative images.


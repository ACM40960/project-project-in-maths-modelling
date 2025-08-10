# Night-time Wildlife Detection using YOLOv8 with Attention Modules and Custom Head

## Table of Contents

1. [Project Overview](#project-overview)
2. [Motivation](#motivation)
3. [Methodology](#methodology)
    - [Data Collection & Preprocessing](#data-collection--preprocessing)
    - [Model Architecture](#model-architecture)
    - [Model Variants Compared](#model-variants-compared)
    - [Training Comfiguration](#training-configuration)
    - [Evaluation Metrics](#evaluation-metrics)
4. [Project Structure](#project-structure)
5. [Results](#results)
6. [How to Run](#how-to-run)
7. [Project Poster](#project-poster)
8. [References](#references)
9. [License](#license)


## Project Overview

This project explores architectural modifications to the YOLOv8 object detection model for improved performance on infrared wildlife images captured at night. Our primary goal is to investigate whether integrating attention mechanisms and custom detection heads can enhance detection accuracy under low-light conditions. 

We implement and evaluate the following model variants: 

- **YOLOv8n**: Baseline model provided by Ultralytics
- **YOLOv11n**: Baseline model provided by Ultralytics
- **YOLOv8n + OCCAPCC(end)**: OCCAPCC attention module appended at the end of the backbone
- **YOLOv8n + OCCAPCC(index 6)**: OCCAPCC inserted at backbone layer 6
- **YOLOv8n + CBAM**: CBAM module used instead of OCCAPCC
- **YOLOv8n + OCCAPCC + Efficent3DBB head**: Combines OCCAPCC with a custom detection head


## Motivation

Accurate recognition of wildlife in their natural habitats is essential for ecological research and biodiversity conservation. While object detection models such as YOLO perform well on well-lit, high-resolution datasets, their performance often degrades in low-light environments or infrared camera settings, which are commonly encountered in real-world field monitoring.

Previous studies have shown that incorporating attention mechanisms can help improve model performance. For example, YOLOv8-night enhances detection accuracy in dark scenes by introducing a channel attention module.

This project aims to investigate whether integrating attention mechanisms and a custom detection head into the YOLOv8 architecture can further improve detection accuracy under low-light conditions.

## Methodology

### Data Collection & Preprocessing

- **Dataset**: We used the *voc_night* subset of the [NTLNP dataset](https://huggingface.co/datasets/myyyyw/NTLNP/tree/main), which contains 10,344 night-time infrared images of wildlife across 17 animals. 

- **Annotation Format**: All annotations follow the Pascal VOC XML format.

- **Preprocessing**

    To prepare the dataset, run the preprocessing scripts under `code` folder: 
    ```
    python preprocessing.py
    ```
    This script will: 
    - Extract class names and save them to `yolo_dataset/classes_names.txt`
    - Convert original annotations to YOLO format
    - Split the dataset into training, validation, and test sets (70% / 10% / 20%)



### Model Architecture
- **Base Model**: 

    The baseline model is YOLOv8n, a lightweight object detection architecture from the Ultralitics framework. We selected it due to its efficiency and low computational cost, which has made it widely adopted in real-time applications such as wildlife monitoring and other resource-constrained scenarios. 

- **YOLOv11n**: 

    YOLOv11n is an improved version officially released by Ultralytics in 2024. Compared to YOLOv8n, it introduces more efficient building blocks such as `c3k2` and `C2PSA`, and replaces the standard detection head with `YOLOEDetect`. These changes allow YOLOv11n to achieve better accuracy with fewer parameters and lower computational cost. 

- **Attention Modules**: 

    - **OCCAPCC** (appended at the end of the backbone): 

        This module is added at the end of the backbone, to refine high-level semantic features before passing them to the detection head. The rationale is that global context and fine-grained spatial cues extracted by OCCAPCC may help improve detection performance in low-contrast, cluttered scenes scenes typical of night-time infrared images. Based on findings from [Wang et al., 2024](#references-3), placing the attention module at the end of the backbone achieves the best performance in low-light conditions.

    - **OCCAPCC** (inserted at index 6): 

        In a separate experiment, OCCAPCC is placed mid-backbone (at index 6) to assess whether earlier feature enhancement leads to better representations. This positioning helps evaluate how the depth of attention integration affects model performance. 

    - **CBAM** (inserted before SPPF): 

        CBAM is placed before SPPF block to enhance deep semantic features before multi-scale pooling. This position allows CBAM to operate on rich, uncompressed feature maps, improving channel and spetial attention. According to [Woo et al., 2018](#references-4), CBAM is most effective when applied after backbone blocks. Placing it before SPPF alighs with this recommentation and avoids disrupting the output used by the neck. 

- **Detection Head**: 

    - **Efficient3DBB Head**: 


### Model Variants Compared
We conducted experiments with four model variants to evaluate the impact of attention mechanisms and custom head design: 

| Model variant | Description | 
| ---- | ---- | 
| Baseline YOLOv8n | Standard YOLOv8n from Ultralytics | 
| Baseline YOLOv11n | Standard YOLOv11n from Ultralytics | 
| YOLOv8n + OCCAPCC (end) | YOLOv8n with `OCCAPCCChannelAttention` at the end of the backbone | 
| YOLOv8n + OCCAPCC (index 6) | YOLOv8n with `OCCAPCCChannelAttention` at layer 6 of the backbone | 
| YOLOv8n + CBAM | YOLOv8n with `CBAM` before `SPPF` | 
| YOLOv8n + OCCAPCC + Efficient3DBB | YOLOv8 with `OCCAPCCChannelAttention` at the end of the backbone and a custom head `Detect_Efficient3DBB` |


### Training Configuration
- **Input Size**: 640 $\times$ 640
- **Batch Size**: 16
- **Epochs**: 50


### Evaluation Metrics

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5. Measures how well the model detects objects with acceptable overlap.
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95 (step 0.05). Stricter and more comprehensive than mAP50.
- **Precision**: The ratio of correctly predicted positives to all predicted positives. Reflects how many detections are correct. 
- **Recall**: The ratio of correctly predicted positives to all actual positives. Reflects how many true objects are detected.


## Project Structure

```
project/
├── code/
│   ├── ultralytics_attention/
│   ├── ultralytics_head/
│   ├── 
│   ├── 
│   └──
├── yolo_dataset/ 
├── results
│   ├── saved_models/
│   │   ├── yolov8_best.pt
│   │   ├── yolov11_best.pt
│   │   ├──
│   │   ├──
│   │   └──
│   └── result.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

## Results

| Model Variant | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 
| ---- | ---- | ---- | ---- | ---- | 
| Baseline YOLOv8n | 
| Baseline YOLOv11n | 
| YOLOv8n + OCCAPCC (end) |
| YOLOv8n + OCCAPCC (index 6) | 
| YOLOv8n + CBAM |  
| YOLOv8n + OCCAPCC + Efficient3DBB | 


## How to Run


## Project Poster

See Project_Poster.pdf for a visual summary of the project.

## References

[1] NTLNP Dataset - https://huggingface.co/datasets/myyyyw/NTLNP

[2] Ultralytics Document - https://github.com/ultralytics/ultralytics

<a id='reference-3'>[3]</a> Tianyu Wang, Siyu Ren, Haiyan Zhang. *Nighttime wildlife object detection based on YOLOv8-night*. Electronics Letters, vol. 60, no. 15, 2024. [https://doi.org/10.1049/ell2.13305](https://doi.org/10.1049/ell2.13305)


<a id='references-4'>[4]</a> Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon. *CBAM: Convolutional Block Attention Module*. arXiv:1807.06521 [cs.CV], 2018. [https://doi.org/10.48550/arXiv.1807.06521](https://doi.org/10.48550/arXiv.1807.06521)


## License




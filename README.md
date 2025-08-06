# Night-time Wildlife Detection using YOLOv8 with Attention Modules and Custom Head

## Table of Contents

1. [Abstract](#abstract)
2. [Project Overview](#project-overview)
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


## Abstract

This project aims to improve object detection performance on night-time infrared wildlife images. We use the *voc_night* subset of the NTLNP dataset, containing 10.344 annotated images across 17 species. Our approach is based on the YOLOv8 and YOLOv11 object detection architectures. We further enhance YOLOv8 by inserting `OCCAPCCChannelAttention` modules after the backbone and replacing the detection head with a `Detect_Efficient3DBB` structure. We compare four model variants under the same training conditions. 

...


## Project Overview


## Methodology

### Data Collection & Preprocessing

- **Dataset**: We used the *voc_night* subset of the [NTLNP dataset](https://huggingface.co/datasets/myyyyw/NTLNP/tree/main), which contains 10.344 night-time infrared images of wildlife across 17 animals. 

- **Annotation Format**: All annotations follow the Pascal VOC XML format.

- **Preprocessing**
    - Converted original annotations to YOLO format. 
    - Split the dataset into training, validation, and test sets. 


### Model Architecture
- **Base Model**: YOLOv8n from Ultralytics framework
- **Custom Components**: 
    - Attention
    - Head


### Model Variants Compared
We conducted experiments with four model variants to evaluate the impact of attention mechanisms and custom head design: 

| Model variant | Description | 
| ---- | ---- | 
| Baseline YOLOv8n | Standard YOLOv8n from Ultralytics | 
| Baseline YOLOv11n | Standard YOLOv11n from Ultralytics | 
| YOLOv8n + Attention | YOLOv8 with `OCCAPCCChannelAttention` attention modules | 
| YOLOv8n + Attention + Head | YOLOv8 with OCC + APCC attention and a custom head `Detect_Efficient3DBB` |


### Training Configuration
- **Input Size**: 640 $\times$ 640
- **Batch Size**: 16
- **Epochs**: 50
- **Loss Components**: 


### Evaluation Metrics



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

| Model Variant | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 
| ---- | ---- | ---- | ---- | ---- | 
| Baseline YOLOv8n | 
| Baseline YOLOv11n | 
| YOLOv8n + Attention | 
| YOLOv8n + Attention + Head | 


## How to Run


## Project Poster

See Project_Poster.pdf for a visual summary of the project.

## References

- NTLNP Dataset - https://huggingface.co/datasets/myyyyw/NTLNP

## License




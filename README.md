

## Table of Contents


## Abstract


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


## How to Run


## Project Poster


## References


## Team members


## License




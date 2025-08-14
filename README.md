# Night-time Wildlife Detection using YOLOv8 with Attention Modules and Custom Head

## Table of Contents



## Abstract

Accurate recognition of wildlife in their natural habitats is essential for ecological research and biodiversity conservation. While object detection models such as YOLO perform well on well-lit, high-resolution datasets, their performance often degreades in low-light environments or infrared camera settings, which are common in rael-world field monitoring. 

This project explores architectural modifications to the YOLOv8 object detection model for improved performance on infrared wildlife images captured at night. Specially, we investigate whether integrating attention mechanisms (OCCAPCC, CBAM) and a custom detection head (Efficient3DBB) can enhance detection accuracy under low-light conditions. We evaluate multiple variants, including YOLOv8n and YOLOv11n baselines, uncer the same training settings. 


## Project Overview

We implement and evaluate the following model variants: 

- **YOLOv8n**: Baseline model provided by Ultralytics
- **YOLOv11n**: Baseline model provided by Ultralytics
- **YOLOv8n + OCCAPCC(end)**: OCCAPCC attention module appended at the end of the backbone
- **YOLOv8n + OCCAPCC(index 6)**: OCCAPCC inserted at backbone layer 6
- **YOLOv8n + CBAM**: CBAM module used instead of OCCAPCC
- **YOLOv8n + OCCAPCC + Efficent3DBB**: Combines OCCAPCC with a custom detection head Efficient3DBB
- **YOLOv8n + CBAM + Efficient3DBB**: Combines CBAM with Efficient3DBB head


## Installation

1. Clone the repository: 

    ``` sh
    git clone https://github.com/ACM40960/project-project-in-maths-modelling
    cd  project-project-in-maths-modelling
    ```

2. Create virtual environments: 

    You need to set up three separate environments for running different model variants: 

    - **Baseline** - official Ultralytics YOLOv8 / YOLOv11
    - **Attention** - modified Ultralytics (located in `code/ultralytics_attention`)
    - **Head** - modified Ultralytics with custom detection head (located in `code/ultralytics_head`)


    1. **Baseline Environment**

        ``` sh
        # create and activate environment
        conda create -n yolo_baseline python=3.9 -y
        conda activate yolo_baseline

        # install dependencies
        pip install -r requirements.txt

        # install official ultralytics
        pip install ultralytics
        ```

    2. **Attention Environment**

        ``` sh
        conda create -n yolo_attention python=3.9 -y
        conda activate yolo_attention

        pip install -r requirements.txt

        cd code/ultralytics_attention
        pip install -e .
        cd ../..
        ```

    3. **Head Environment**

        ``` sh
        conda create -n yolo_head python=3.9 -y
        conda activate yolo_head

        pip install -r requirements.txt

        cd code/ultralytics_head
        pip install -e .
        cd ../..
        ```
3. Download the dataset: 

    - Get `voc_night.rar` from the [NTLNP dataset](https://huggingface.co/datasets/myyyyw/NTLNP)
    - Extract the archive directly into the project directory so that the extracted folder is named `voc_night`:

        ```plaintext
        project/
        ├── voc_night/
        │   ├── JPEGImages/
        │   └── Annotations/
        ```

4. Preprocess the dataset:
    
    ``` sh
    cd code
    python preprocessing.py
    ```

    This will create a new folder `yolo_dataset` under the project directory and perform the following steps: 
    
    - Extract class names and save them to `yolo_dataset/classes_names.txt` 
    - Convert the original annotations from Pascal VOC XML to YOLO format 
    - Split the dataset into training, validation, and test sets with a **70% / 10% / 20%** ratio 
    - Generate `yolo_dataset/data.yaml` for YOLO training 
    
    Resulting structure:

    ``` plaintext
    project/
    ├── voc_night/              # original dataset
    │   ├── JPEGImages/
    │   ├── Annotations/
    │   └── YOLOLabels/         # temporary YOLO-format labels
    ├── yolo_dataset/           # processed dataset for YOLO training
    │   ├── images/
    │   │   ├── train
    │   │   ├── valid
    │   │   └── test
    │   ├── labels/
    │   │   ├── train
    │   │   ├── valid
    │   │   └── test
    │   ├── data.yaml           # YOLO dataset configuration file
    │   └── classes_names.txt   # list of all class names
    ```

5. Running training and validation

    - **Baseline Environment** (`yolo_baseline`)

        ``` sh
        conda activate yolo_baseline
        cd code
        python yolo+baseline_train.py
        python yolo+baseline_val_pred.py
        ```

    - **Attention Environment** (`yolo_attention`)

        ``` sh
        conda activate yolo_attention
        cd code
        python yolo+attention_train.py
        python yolo+attention_val_pred.py
        ```

    - **Head Environment** (`yolo_head`)

        ``` sh
        conda activate yolo_head
        cd code
        python yolo+head_train.py
        python yolo+head_val_pred.py
        ```

    This will train the corresponding models, run validation on the test split, generate predictions, and save the following outputs under the `results` directory:

    - Best weights (`results/saved_models/*_best.pt`)
    - Training logs and artefacts (`results/*_train/`)
    - Validation results (`results/*_val/`)
    - Prediction results (`results/*_pred/`)
    - Evaluation metrics (`results/*.json`)

6. Analyze results

    Open `results/result.ipynb` to aggregate all JSON metrics into a comparison table.

## Project Structure

``` plaintext
project/ 
├── code/                                       # source code    
│   ├── ultralytics_attention/                  # modified Ultralytics with attention modules
│   │   ├── ultralytics/                        # Ultralytics source code (attention version) 
│   │   ├── ...                                 # other supporting files 
│   │   ├── yolov8+OCCAPCC.yaml                 # YOLO config with OCCAPCC at backbone end
│   │   ├── yolov8+OCCAPCC_index6.yaml          # YOLO config with OCCAPCC at backbone layer 6 
│   │   └── yolov8+CBAM.yaml                    # YOLO config with CBAM 
│   ├── ultralytics_head/                       # modified Ultralytics with custom detect head
│   │   ├── ultralytics/                        # Ultralytics source code (head version)
│   │   ├── ... 
│   │   ├── yolov8+OCCAPCC+Efficient3dbb.yaml   # YOLO config with OCCAPCC + Efficient3DBB
│   │   └── yolov8+CBAM+Efficient3dbb.yaml      # YOLO config with CBAM + Efficient3DBB
│   ├── yolov8n.pt                              # pretrained YOLOv8n weights
│   ├── yolo11n.pt                              # pretrained YOLOv11n weights
│   ├── preprocessing.py                        # script for dataset preprocessing
│   ├── yolo+baseline_train.py                  # training script for YOLOv8n & YOLOv11n baselines
│   ├── yolo+baseline_val_pred.py               # validation & prediction script for baselines
│   ├── yolo+attention_train.py                 # training script for attention models
│   ├── yolo+attention_val_pred.py              # validation & prediction script for attention models 
│   ├── yolo+head_train.py                      # training script for head models
│   └── yolo+head_val_pred.py                   # valiation & prediction script for head models
├── results/                                    # model outputs and evaluation results
│   ├── saved_models/                           # best weights from training
│   │   └── *_best.pt 
│   ├── *_val/                                  # validation results (per model)
│   ├── *.json                                  # evaluation results (per model)
│   └── result.ipynb                            # notebook for aggregating metrics 
├── images/                                     # images for README
├── README.md                                   # project documentation
├── requirements.txt                            # Python dependencies
└── Literature_review.pdf                       # literature review document
```


## Methodology

### Data Collection and Preprocessing

- **Dataset**: 
    
    *voc_night* subset of the NTLNP dataset containing 10,344 night-time infrared images across 17 animal classes. 

- **Annotation Format**: 

    Pascal VOC XML format. 

- **Preprocessing Steps**: 

    Converted to YOLO format, split into train/val/test sets (70% / 10% / 20%), and generated data.yaml for YOLO training *(see [Installation – Step 4](#installation) for execution details)*.

### Model Architecture

- **Baselines**: YOLOv8n, YOLOv11n (official Ultralytics)
- **Attention Models**: YOLOv8n with OCCAPCC (end / index 6) or CBAM
- **Head Models**: YOLOv8n with OCCAPCC or CBAM combined with Efficient3DBB

Full arthitecture definitions are available in: 

- [Attention Models YAMLs](code/ultralytics_attention/)
- [Head Models YAMLs](code/ultralytics_head/)

![Model Architecture](images/model_architecture.png)
*Figure: YOLOv8n backbone with OCCAPCC attention and Efficient3DBB detection head.*


### Trainign Configuration

- **Input Size**: 640 $\times$ 640
- **Batch Size**: 16
- **Epochs**: 50


### Evaluation Metrics

- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.5:0.95**: Averaged mAP across IoU thresholds 0.5 ~ 0.95
- **Precision**: Correct detections among all predicted positives
- **Recall**: Correct detections among all actual positives

## Results

| Model Variant | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 
| ---- | ---- | ---- | ---- | ---- | 
| Baseline YOLOv8n | 
| Baseline YOLOv11n | 
| YOLOv8n + OCCAPCC (end) |
| YOLOv8n + OCCAPCC (index 6) | 
| YOLOv8n + CBAM |  
| YOLOv8n + OCCAPCC + Efficient3DBB | 
| YOLOv8n + CBAM + Efficient3DBB | 


## Project Poster

See [Project_Poster.pdf]() for a visual summary of the project.


## References

[1] NTLNP Dataset - https://huggingface.co/datasets/myyyyw/NTLNP

[2] Ultralytics Document - https://github.com/ultralytics/ultralytics

[3] Tianyu Wang, Siyu Ren, Haiyan Zhang. *Nighttime wildlife object detection based on YOLOv8-night*. Electronics Letters, vol. 60, no. 15, 2024. [https://doi.org/10.1049/ell2.13305](https://doi.org/10.1049/ell2.13305)


[4] Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon. *CBAM: Convolutional Block Attention Module*. arXiv:1807.06521 [cs.CV], 2018. [https://doi.org/10.48550/arXiv.1807.06521](https://doi.org/10.48550/arXiv.1807.06521)

[5] Wan, D., Lu, R., Hu, B., Yin, J., Shen, S., Xu, T., & Lang, X. (2024). *YOLO-MIF: Improved YOLOv8 with Multi-Information fusion for object detection in gray-scale images.* Advanced Engineering Informatics, 62(B), 102709. [https://doi.org/10.1016/j.aei.2024.102709](https://doi.org/10.1016/j.aei.2024.102709)


## License




















## Methodology

### Data Collection & Preprocessing

- **Dataset**: We used the *voc_night* subset of the [NTLNP dataset](https://huggingface.co/datasets/myyyyw/NTLNP/tree/main), which contains 10,344 night-time infrared images of wildlife across 17 animals. 

- **Annotation Format**: All annotations follow the Pascal VOC XML format.

- **Preprocessing**

    Before running the preprocessing script, download `voc_night.rar` from the [NTLNP dataset](https://huggingface.co/datasets/myyyyw/NTLNP/tree/main) and Extract the archive into the `voc_night` folder in your project directory.

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
| YOLOv8n + CBAM + Efficient3DBB | YOLOv8 with `CBAM` in the backbone and a custom detection head `Detect_Efficient3DBB` |


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
│   │   ├── ultralytics/
│   │   ├── ...
│   │   ├── yolov8+OCCAPCC.yaml
│   │   ├── yolov8+OCCAPCC_index6.yaml
│   │   └── yolov8+CBAM.yaml
│   │ 
│   ├── ultralytics_head/
│   │   ├── ultralytics/
│   │   ├── ...
│   │   ├── yolov8+OCCAPCC+Efficient3dbb.yaml
│   │   └── yolov8+CBAM+Efficient3dbb.yaml
│   │ 
│   ├── yolov8n.pt
│   ├── yolo11n.pt
│   ├── preprocessing.py
│   ├── yolo+baseline_train.py
│   ├── yolo+baseline_val.py
│   ├── yolo+attention_train.py
│   ├── yolo+attention_val.py
│   ├── yolo+head_train.py
│   └── yolo+head_val.py
│   
├── results/
│   ├── saved_models/
│   │   └── *_best.pt
│   │
│   ├── *.json
│   └── result.ipynb
│
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
| YOLOv8n + CBAM + Efficient3DBB | 


## How to Run


## Project Poster

See Project_Poster.pdf for a visual summary of the project.

## References

[1] NTLNP Dataset - https://huggingface.co/datasets/myyyyw/NTLNP

[2] Ultralytics Document - https://github.com/ultralytics/ultralytics

<a id='reference-3'>[3]</a> Tianyu Wang, Siyu Ren, Haiyan Zhang. *Nighttime wildlife object detection based on YOLOv8-night*. Electronics Letters, vol. 60, no. 15, 2024. [https://doi.org/10.1049/ell2.13305](https://doi.org/10.1049/ell2.13305)


<a id='references-4'>[4]</a> Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon. *CBAM: Convolutional Block Attention Module*. arXiv:1807.06521 [cs.CV], 2018. [https://doi.org/10.48550/arXiv.1807.06521](https://doi.org/10.48550/arXiv.1807.06521)

[5] Wan, D., Lu, R., Hu, B., Yin, J., Shen, S., Xu, T., & Lang, X. (2024). *YOLO-MIF: Improved YOLOv8 with Multi-Information fusion for object detection in gray-scale images.* Advanced Engineering Informatics, 62(B), 102709. [https://doi.org/10.1016/j.aei.2024.102709](https://doi.org/10.1016/j.aei.2024.102709)

## License




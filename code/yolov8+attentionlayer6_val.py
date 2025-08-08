import json
from ultralytics import YOLO
import torch

name = 'attention_layer6'
model_path = '../results/saved_models/yolov8+attention_layer6_best.pt'
result_path = '../results/' + name + '_result.json'

print(f'Loading model: {model_path}')
model = YOLO(model_path)

print(f'Running validation on {name}')
results = model.val(data = '../yolo_dataset/data.yaml', split = 'test')

metrics = {
    'model': name, 
    'mAP50': round(results.box.map50, 4), 
    'mAP50_95': round(results.box.map, 4), 
    'precision': round(results.box.mp, 4), 
    'recall': round(results.box.mr, 4)
}

with open(result_path, 'w') as f: 
    json.dump(metrics, f, indent = 4)

print(f'Results saved to {result_path}')
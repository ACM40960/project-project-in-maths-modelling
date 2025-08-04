# imports
from ultralytics import YOLO
import yaml
import os


# define paths
dataset_dir = '../yolo_dataset'
data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
model_save_path = '../results/saved_models/yolov11_best.pt'

os.makedirs(os.path.dirname(model_save_path), exist_ok=True)


# train YOLOv11
model = YOLO('yolo11n.pt')

results = model.train(
    data = data_yaml_path, 
    epochs = 50, 
    imgsz = 640, 
    batch = 16
)

trained_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')

if os.path.exists(trained_model_path):
    os.system(f'cp {trained_model_path} {model_save_path}')
    print(f"Model saved to {model_save_path}")
else:
    print("Could not find trained model at:", trained_model_path)
from ultralytics import YOLO
import os


dataset_dir = '../yolo_dataset'
data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
custom_yaml_path = 'ultralytics_head/yolov8n+CBAM+Efficient3dbb.yaml'
model_save_path = '../results/saved_models/yolov8+CBAM+Efficient3dbb_best.pt'

save_model = True

os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

model = YOLO(custom_yaml_path)

results = model.train(
    data=data_yaml_path,
    epochs=50, 
    imgsz=640,
    batch=16
)

if save_model: 
    trained_model_path = os.path.join(model.trainer.save_dir, 'weights', 'best.pt')
    if os.path.exists(trained_model_path):
        os.system(f'cp {trained_model_path} {model_save_path}')
        print(f"Model saved to {model_save_path}")
    else:
        print("Could not find trained model at:", trained_model_path)
else:
    print("Model saving skipped.")
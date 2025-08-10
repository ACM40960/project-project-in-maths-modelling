from ultralytics import YOLO
import os
import shutil

data_yaml = '../yolo_dataset/data.yaml'
model_save_path = '../results/saved_models'
os.makedirs(model_save_path, exist_ok=True)

models_info = {
    'v8+OCCAPCC': {
        'init_weights': 'ultralytics_attention/yolov8+OCCAPCC.yaml', 
        'save_path': os.path.join(model_save_path, 'yolov8+OCCAPCC_best.pt')
    }, 
    'v8+OCCAPCC_index6': {
        'init_weights': 'ultralytics_attention/yolov8+OCCAPCC_index6.yaml', 
        'save_path': os.path.join(model_save_path, 'yolov8+OCCAPCC_index6_best.pt')
    }, 
    'v8+CBAM': {
        'init_weights': 'ultralytics_attention/yolov8+CBAM.yaml', 
        'save_path': os.path.join(model_save_path, 'yolov8+CBAM_best.pt')
    }
}

for name, cfg in models_info.items(): 
    print(f'\n--- Training model: {name} ---\n')

    model = YOLO(cfg['init_weights'])
    results = model.train(
        data = data_yaml, 
        epochs = 50, 
        imgsz = 640, 
        batch = 16
    )

    trained_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    if os.path.exists(trained_model_path):
        os.makedirs(os.path.dirname(cfg['save_path']), exist_ok=True)
        shutil.copy2(trained_model_path, cfg['save_path'])
        print(f"Model saved to {cfg['save_path']}")
    else:
        print(f"Could not find trained model at: {trained_model_path}")

    train_folder = results.save_dir
    if train_folder:
        target_folder = os.path.join('..', 'results', f'{name}_train')
        if os.path.exists(target_folder):
            shutil.rmtree(target_folder)
        shutil.move(train_folder, target_folder)
        print(f"Moved {train_folder} → {target_folder}")
    else:
        print("No train output folder found.")
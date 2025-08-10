import json
from ultralytics import YOLO
import os
import shutil

# model informations 
models_info = {
    'v8baseline': '../results/saved_models/yolov8_best.pt', 
    'v11baseline': '../results/saved_models/yolov8v11_best.pt'
}

data_yaml = '../yolo_dataset/data.yaml'
runs_base = 'runs/detect'

# get the latest validation run folder
def get_latest_val_folder(base = runs_base): 
    folders = [f for f in os.listdir(base) if f.startswith('val')]
    folders = sorted(folders, key=lambda x: int(x.replace('val', '')) if x.replace('val', '').isdigit() else -1)
    return os.path.join(base, folders[-1]) if folders else None

for name, model_path in models_info.items():

    print(f'\n--- Validating model: {name} ---\n')

    model = YOLO(model_path)
    results = model.val(data = data_yaml, split = 'test')

    val_folder = get_latest_val_folder()
    if val_folder is None:
        raise RuntimeError("No val output folder found.")

    target_folder = os.path.join('..', 'results', f'{name}_val')
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    shutil.move(val_folder, target_folder)
    print(f'Moved {val_folder} → {target_folder}')

    # store metrics
    metrics = {
        'model': name,
        'mAP50': round(results.box.map50, 4),
        'mAP50_95': round(results.box.map, 4),
        'precision': round(results.box.mp, 4),
        'recall': round(results.box.mr, 4)
    }
    json_path = os.path.join('..', 'results', f'{name}.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f'Metrics saved to {json_path}')
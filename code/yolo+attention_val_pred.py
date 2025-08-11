from ultralytics import YOLO
import json
import os
import shutil
import yaml

do_val = True   # whether to run val
do_pred = True  # whether to run predict

# model informations 
models_info = {
    'v8+OCCAPCC': '../results/saved_models/yolov8+OCCAPCC_best.pt', 
    'v8+OCCAPCC(index6)': '../results/saved_models/yolov8+OCCAPCC_index6_best.pt',
    'v8+CBAM': '../results/saved_models/yolov8+CBAM_best.pt'
}

data_yaml = '../yolo_dataset/data.yaml'
runs_base = 'runs/detect'
conf_threshold = 0.25

def get_latest_folder(prefix: str, base: str = runs_base): 
    if not os.path.isdir(base): 
        return None
    folders = [f for f in os.listdir(base) if f.startswith(prefix)]
    if not folders: 
        return None
    folders = sorted(
        folders, 
        key = lambda x: int(x.replace(prefix, '')) if x.replace(prefix, '').isdigit() else 0
    )
    return os.path.join(base, folders[-1])

def move_dir(src: str, dst: str): 
    if os.path.exists(dst): 
        shutil.rmtree(dst)  # delete target if exists
    shutil.move(src, dst)
    print(f'Moved {src} -> {dst}')

for name, model_path in models_info.items():
    print(f'\n ===== Model: {name} ===== \n')
    model = YOLO(model_path)

    # val
    if do_val: 
        print(f'[VAL] {name}\n')

        results = model.val(data = data_yaml, split = 'test')
        val_folder = get_latest_folder('val')
        if val_folder is None:
            raise RuntimeError("No val output folder found.")
        target_val = os.path.join('..', 'results', f'{name}_val')
        move_dir(val_folder, target_val)

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

    
    # predict
    if do_pred: 
        print(f'[PREDICT] {name}\n')

        # read test folder from data.yaml
        with open(data_yaml, 'r') as f: 
            data_cfg = yaml.safe_load(f)
        images_dir = data_cfg.get('test')
        if not images_dir: 
            raise ValueError("'test' folder not found in data.yaml")
        if not os.path.isabs(images_dir): 
            images_dir = os.path.join(os.path.dirname(data_yaml), images_dir)

        model.predict(source = images_dir, save = True, conf = conf_threshold)
        pred_folder = get_latest_folder('predict')
        if pred_folder is None:
            raise RuntimeError("No predict output folder found.")
        target_pred = os.path.join('..', 'results', f'{name}_pred')
        move_dir(pred_folder, target_pred)
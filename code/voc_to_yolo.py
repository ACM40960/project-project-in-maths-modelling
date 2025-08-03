import os
import xml.etree.ElementTree as ET

# 資料夾路徑
voc_dir = '../data/voc_night'
image_dir = os.path.join(voc_dir, 'JPEGImages')
annot_dir = os.path.join(voc_dir, 'Annotations')
yolo_label_dir = os.path.join(voc_dir, 'YoloLabels')
os.makedirs(yolo_label_dir, exist_ok=True)

# 讀取 class names
with open('../data/classes_names.txt', 'r') as f: 
    classes = [line.strip() for line in f if line.strip()]

# 開始轉換
for xml_file in os.listdir(annot_dir): 
    if not xml_file: 
        continue
    
    tree = ET.parse(os.path.join(annot_dir, xml_file))
    root = tree.getroot()

    filename = root.find('filename').text
    img_path = os.path.join(image_dir, filename)
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    yolo_lines = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # 轉換為 YOLO 格式
        x_center = (xmin + xmax) / 2.0 / w
        y_center = (ymin + ymax) / 2.0 / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 輸出 .txt 標註檔
    txt_filename = os.path.splitext(filename)[0] + '.txt'
    with open(os.path.join(yolo_label_dir, txt_filename), 'w') as f:
        f.write('\n'.join(yolo_lines))

print("Finish")
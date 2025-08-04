import os
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split

# 原始資料夾
dir_path = './voc_night'
image_dir = os.path.join(dir_path, 'JPEGImages')
annot_dir = os.path.join(dir_path, 'Annotations')
label_dir = os.path.join(dir_path, 'labels')
os.makedirs(label_dir, exist_ok = True)

# 新資料夾
output_dir = './yolo_dataset'
for split in ['train', 'test']: 
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# 類別名稱
classes = [
    "AmurTiger",
    "Badger",
    "BlackBear",
    "Cow",
    "Dog",
    "Hare",
    "Leopard",
    "LeopardCat",
    "MuskDeer",
    "RaccoonDog",
    "RedFox",
    "RoeDeer",
    "Sable",
    "SikaDeer",
    "Weasel",
    "WildBoar",
    "Y.T.Marten"
]

# 轉換 xml annotations 成 yolo txt
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
    with open(os.path.join(label_dir, txt_filename), 'w') as f:
        f.write('\n'.join(yolo_lines))


# 分割
images = []
labels = []

for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue

    txt_path = os.path.join(label_dir, label_file)
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        continue  # 跳過空標籤檔

    class_id = lines[0].strip().split()[0]  # 第一句的 class_id
    labels.append(class_id)

    img_name = label_file.replace(".txt", ".jpg")
    images.append(img_name)

# 分層切分訓練與測試
train_imgs, test_imgs = train_test_split(
    images, test_size=0.2, stratify=labels, random_state=123
)

# 複製檔案到對應目錄
def copy_to_split(image_list, split):
    for img_file in image_list:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".txt"

        # 圖片
        shutil.copy(
            os.path.join(image_dir, img_file),
            os.path.join(output_dir, "images", split, img_file)
        )
        # 標籤
        shutil.copy(
            os.path.join(label_dir, label_file),
            os.path.join(output_dir, "labels", split, label_file)
        )

# 執行複製
copy_to_split(train_imgs, "train")
copy_to_split(test_imgs, "test")

print('Finish spliting')



# 建立 data.yaml
data_yaml_path = os.path.join(output_dir, "data.yaml")

with open(data_yaml_path, "w") as f:
    f.write("train: images/train\n")
    f.write("val: images/test\n\n")
    f.write(f"nc: {len(classes)}\n")
    f.write("names: [")
    f.write(", ".join(f"'{cls}'" for cls in classes))
    f.write("]\n")

print(f"✅ Generated data.yaml at {data_yaml_path}")

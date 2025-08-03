import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# original path
base_dir = "../data/voc_night"
image_dir = os.path.join(base_dir, "JPEGImages")
enhanced_dir = os.path.join(base_dir, "JPEGImagesEnhanced")
label_dir = os.path.join(base_dir, "YoloLabels")

# new path
output_dir = "../data/split_voc"
for split in ['train', 'test']:
    os.makedirs(os.path.join(output_dir, split, "JPEGImages"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "JPEGImagesEnhanced"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "YoloLabels"), exist_ok=True)

# get label names
images = []
labels = []

for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue
    
    txt_path = os.path.join(label_dir, label_file)
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        continue

    first_line = lines[0].strip()
    class_id = first_line.split()[0]
    labels.append(class_id)

    img_name = label_file.replace(".txt", ".jpg")
    images.append(img_name)


# 分割（stratified）
train_imgs, test_imgs = train_test_split(
    images, test_size=0.2, stratify=labels, random_state=123
)

def copy_to_split(image_list, split):
    for img_file in image_list:
        name = os.path.splitext(img_file)[0]
        label_file = name + ".txt"

        shutil.copy(os.path.join(image_dir, img_file), os.path.join(output_dir, split, "JPEGImages", img_file))
        shutil.copy(os.path.join(enhanced_dir, img_file), os.path.join(output_dir, split, "JPEGImagesEnhanced", img_file))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(output_dir, split, "YoloLabels", label_file))


copy_to_split(train_imgs, "train")
copy_to_split(test_imgs, "test")

print('Finish')
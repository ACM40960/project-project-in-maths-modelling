import os
import shutil
from sklearn.model_selection import train_test_split

# original path
base_dir = "../data/voc_night"
image_dir = os.path.join(base_dir, "JPEGImages")
label_dir = os.path.join(base_dir, "YoloLabels")

# new YOLO-format path
output_dir = "../data/yolo_dataset"
for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

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

# split train, valid, test as 70%, 10%, 20%
train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
    images, labels, test_size=0.3, stratify=labels, random_state=123
)

valid_imgs, test_imgs, _, _ = train_test_split(
    temp_imgs, temp_labels, test_size=2/3, stratify=temp_labels, random_state=123
)

def copy_to_split(image_list, split):
    for img_file in image_list:
        name = os.path.splitext(img_file)[0]
        label_file = name + ".txt"

        shutil.copy(os.path.join(image_dir, img_file), os.path.join(output_dir, "images", split, img_file))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(output_dir, "labels", split, label_file))

copy_to_split(train_imgs, "train")
copy_to_split(valid_imgs, "valid")
copy_to_split(test_imgs, "test")

print("Finish")

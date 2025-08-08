import os
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split

# original path
dir_path = '../voc_night/'
image_dir = os.path.join(dir_path, 'JPEGImages')
annot_dir = os.path.join(dir_path, 'Annotations')
label_dir = os.path.join(dir_path, 'YoloLabels')
os.makedirs(label_dir, exist_ok=True)

# output path
output_dir = '../yolo_dataset'
for split in ['train', 'valid', 'test']: 
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

# get class names
def get_classes(): 
    """Collect class names from XML annotations and save to a file."""

    print("Collecting class names...")

    classes = set()

    for file in os.listdir(annot_dir):
        if file.endswith('.xml'):
            tree = ET.parse(os.path.join(annot_dir, file))
            root = tree.getroot()

            for obj in root.findall('object'):
                name = obj.find('name').text
                classes.add(name)
    
    # save class names to a txt file
    output_path = os.path.join(output_dir, 'classes_names.txt')
    with open(output_path, 'w') as f: 
        for cls in sorted(classes): 
            f.write(cls + '\n')

    print(f"Classes saved to {output_path}")
    return sorted(classes)

# convert xml annotations to yolo txt
def convert_annotations(classes):
    """Convert XML annotations to YOLO format and save to label directory."""

    print("Converting annotations to YOLO format...")

    for xml_file in os.listdir(annot_dir): 
        if not xml_file.endswith('.xml'):
            continue
        tree = ET.parse(os.path.join(annot_dir, xml_file))
        root = tree.getroot()

        # get image filename and check if it exists
        filename = root.find('filename').text
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        # get image size
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        # convert annotations to YOLO format
        yolo_lines = []
        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)

            # get bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # convert to YOLO format
            x_center = (xmin + xmax) / 2.0 / w
            y_center = (ymin + ymax) / 2.0 / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h

            # create YOLO format line
            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # output .txt annotation file
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        with open(os.path.join(label_dir, txt_filename), 'w') as f:
            f.write('\n'.join(yolo_lines))

    print("Annotations converted finished.")

# copy images and labels to respective split directories
def copy_to_split(image_list, split):
    """Copy images and labels to the respective split directories."""
    
    for img_file in image_list:
        name = os.path.splitext(img_file)[0]
        label_file = name + ".txt"
        
        # copy image and label files
        shutil.copy(os.path.join(image_dir, img_file), os.path.join(output_dir, "images", split, img_file))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(output_dir, "labels", split, label_file))

# split dataset into train, valid, test
def split_dataset():
    """Split dataset into train, valid, and test sets."""

    print("Splitting dataset into train, valid, and test sets...")
    
    images = []
    labels = []

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        # read label file
        txt_path = os.path.join(label_dir, label_file)
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            continue
            
        # get class id from the first line
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

    # copy images and labels to respective split directories
    copy_to_split(train_imgs, "train")
    copy_to_split(valid_imgs, "valid") 
    copy_to_split(test_imgs, "test")

    print("Dataset split finished.")

if __name__ == '__main__': 
    all_classes = get_classes()
    convert_annotations(all_classes)
    split_dataset()
    print("All preprocessing steps completed successfully.")
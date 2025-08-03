import os
import xml.etree.ElementTree as ET

# 資料夾路徑
voc_dir = '../data/voc_night'
annot_dir = os.path.join(voc_dir, 'Annotations')

# 收集類別名稱
all_classes = set()

for file in os.listdir(annot_dir): 
    if file.endswith('.xml'): 
        tree = ET.parse(os.path.join(annot_dir, file))
        root = tree.getroot()

        for obj in root.findall('object'): 
            name = obj.find('name').text
            all_classes.add(name)

# 輸出 classes
output_path = '../data/classes_names.txt'
with open(output_path, 'w') as f:
    for cls in sorted(all_classes):
        f.write(cls + '\n')

print('Finish')
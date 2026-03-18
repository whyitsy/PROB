import os
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import xml.etree.ElementTree as ET
from pycocotools.coco import COCO

# “多线程 + 分块 + 文本一次性写入”


def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


def build_xml_bytes(coco_instance, image_id):
    image_details = coco_instance.imgs[image_id]

    annotation_el = ET.Element('annotation')
    ET.SubElement(annotation_el, 'filename').text = image_details['file_name']

    size_el = ET.SubElement(annotation_el, 'size')
    ET.SubElement(size_el, 'width').text = str(image_details['width'])
    ET.SubElement(size_el, 'height').text = str(image_details['height'])
    ET.SubElement(size_el, 'depth').text = '3'

    anns = coco_instance.imgToAnns.get(image_id, [])
    for annotation in anns:
        object_el = ET.SubElement(annotation_el, 'object')
        ET.SubElement(object_el, 'name').text = coco_instance.cats[annotation['category_id']]['name']
        ET.SubElement(object_el, 'difficult').text = '0'

        x, y, w, h = annotation['bbox']
        bb_el = ET.SubElement(object_el, 'bndbox')
        ET.SubElement(bb_el, 'xmin').text = str(int(x + 1.0))
        ET.SubElement(bb_el, 'ymin').text = str(int(y + 1.0))
        ET.SubElement(bb_el, 'xmax').text = str(int(x + w + 1.0))
        ET.SubElement(bb_el, 'ymax').text = str(int(y + h + 1.0))

    xml_bytes = ET.tostring(annotation_el, encoding='utf-8')
    out_name = image_details['file_name'].rsplit('.', 1)[0] + '.xml'
    return out_name, xml_bytes


def write_one_xml(args):
    target_ann_dir, file_name, xml_bytes = args
    out_path = os.path.join(target_ann_dir, file_name)
    with open(out_path, 'wb') as f:
        f.write(xml_bytes)


def coco_to_voc_detection_optimized(coco_annotation_file, target_folder, num_workers=8, chunk_size=2000):
    target_ann_dir = os.path.join(target_folder, 'Annotations')
    os.makedirs(target_ann_dir, exist_ok=True)

    coco_instance = COCO(coco_annotation_file)
    image_ids = list(coco_instance.imgToAnns.keys())

    total = len(image_ids)
    processed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for chunk in chunked(image_ids, chunk_size):
            # 先在主线程/少量逻辑中构造任务数据，避免一次性占太多内存
            tasks = []
            for image_id in chunk:
                file_name, xml_bytes = build_xml_bytes(coco_instance, image_id)
                tasks.append((target_ann_dir, file_name, xml_bytes))

            list(executor.map(write_one_xml, tasks))

            processed += len(chunk)
            print(f'Processed {processed}/{total} images.')


def imagesets_optimized(coco_train_annotation_file, coco_val_annotation_file, target_folder):
    os.makedirs(os.path.join(target_folder, 'ImageSets'), exist_ok=True)

    coco_train_instance = COCO(coco_train_annotation_file)
    coco_val_instance = COCO(coco_val_annotation_file)

    train_names = [
        coco_train_instance.imgs[image_id]['file_name'].rsplit('.', 1)[0]
        for image_id in coco_train_instance.imgToAnns
    ]
    val_names = [
        coco_val_instance.imgs[image_id]['file_name'].rsplit('.', 1)[0]
        for image_id in coco_val_instance.imgToAnns
    ]

    all_names = train_names + val_names
    out_path = os.path.join(target_folder, 'ImageSets', 'train.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_names) + '\n')



if __name__ == '__main__':
    if len(sys.argv) != 4:
        coco_train_annotation_file = '/mnt/data/kky/datasets/owdetr/data/coco/annotations/instances_train2017.json'
        coco_val_annotation_file = '/mnt/data/kky/datasets/owdetr/data/coco/annotations/instances_val2017.json'
        target_folder = '/mnt/data/kky/datasets/owdetr/data/OWOD'
    else:
        coco_train_annotation_file = sys.argv[1]
        coco_val_annotation_file = sys.argv[2]
        target_folder = sys.argv[3]

    coco_to_voc_detection_optimized(coco_train_annotation_file, target_folder)
    coco_to_voc_detection_optimized(coco_val_annotation_file, target_folder)
    imagesets_optimized(coco_train_annotation_file, coco_val_annotation_file, target_folder)
    

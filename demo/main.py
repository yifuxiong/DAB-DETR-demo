import os
import json
import cv2 as cv
import random

anno_root = 'D:/coco'
# anno_root = '/home/yifux/coco'
image_root = anno_root + '/' + 'val2017'


def parse_anno():
    image = cv.imread('000000289343.jpg')
    anno_path = 'label_sample.json'
    with open(anno_path, 'r', encoding='gbk') as f:
        root = json.load(f)
        annotations = root['annotations']
        for annotation in annotations:
            # image_id = annotation['image_id']
            bbox = annotation['bbox']
            # category_id = annotation['category_id']

            x, y, w, h = bbox
            image = cv.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color=(0, 255, 0), thickness=2)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_images_list(root):
    total_nums = 0
    images_list = []
    for file in os.listdir(root):
        if not file.endswith('jpg'):
            continue
        total_nums += 1
        images_list.append(file)
    return total_nums, images_list


def split_dataset(root, traintest_rate=0.2, trainval_rate=0.2):
    total_nums, images_list = get_images_list(root)
    train_nums = int(total_nums * (1 - traintest_rate))
    test_nums = int(total_nums * traintest_rate)
    val_nums = int(train_nums * trainval_rate)
    print('total_nums:{}, train_nums:{}, test_nums:{}, val_nums:{}'.format(total_nums, train_nums, test_nums, val_nums))
    test_indices = random.sample(images_list, k=test_nums)

    train_point = open('train.txt', 'w', encoding='utf-8')
    val_point = open('val.txt', 'w', encoding='utf-8')
    test_point = open('test.txt', 'w', encoding='utf-8')

    train_indices = []
    for file in os.listdir(root):
        if not file.endswith('jpg'):
            continue
        if file in test_indices:
            test_point.write(file + '\n')
        else:
            train_indices.append(file)

    val_indices = random.sample(train_indices, k=val_nums)
    for file in train_indices:
        if file in val_indices:
            val_point.write(file + '\n')
        else:
            train_point.write(file + '\n')
    print('train, test, val split into files.')


def get_annotations(anno_root):
    save_train = anno_root + '/' + 'annotations_instances_train2017.json'
    save_test = anno_root + '/' + 'annotations_instances_test2017.json'
    save_val = anno_root + '/' + 'annotations_instances_val2017.json'
    context_train = {}
    context_test = {}
    context_val = {}
    imgs_train = []
    imgs_test = []
    imgs_val = []
    annos_train = []
    annos_test = []
    annos_val = []

    train_list = []
    test_list = []
    val_list = []
    with open('train.txt', 'r', encoding='utf-8') as f:
        contents = f.readlines()
        for content in contents:
            train_list.append(content.split('.')[0])
    with open('test.txt', 'r', encoding='utf-8') as f:
        contents = f.readlines()
        for content in contents:
            test_list.append(content.split('.')[0])
    with open('val.txt', 'r', encoding='utf-8') as f:
        contents = f.readlines()
        for content in contents:
            val_list.append(content.split('.')[0])

    anno_path = anno_root + '/' + 'instances_val2017.json'
    with open(anno_path, 'r', encoding='gbk') as f:
        root = json.load(f)

        images = root['images']
        for image in images:
            file_name = image['file_name']
            height = image['height']
            width = image['width']
            id = image['id']

            img_train = {}
            img_test = {}
            img_val = {}

            if file_name.split('.')[0] in test_list:
                img_test['file_name'] = file_name
                img_test['height'] = height
                img_test['width'] = width
                img_test['id'] = id
                imgs_test.append(img_test)
            elif file_name.split('.')[0] in val_list:
                img_val['file_name'] = file_name
                img_val['height'] = height
                img_val['width'] = width
                img_val['id'] = id
                imgs_val.append(img_val)
            else:
                img_train['file_name'] = file_name
                img_train['height'] = height
                img_train['width'] = width
                img_train['id'] = id
                imgs_train.append(img_train)

        annotations = root['annotations']
        for annotation in annotations:
            area = annotation['area']
            iscrowd = annotation['iscrowd']
            image_id = annotation['image_id']
            bbox = annotation['bbox']
            category_id = annotation['category_id']
            id = annotation['id']

            anno_train = {}
            anno_test = {}
            anno_val = {}

            if str(image_id).zfill(12) in test_list:
                anno_test['area'] = area
                anno_test['iscrowd'] = iscrowd
                anno_test['image_id'] = image_id
                anno_test['bbox'] = bbox
                anno_test['category_id'] = category_id
                anno_test['id'] = id
                annos_test.append(anno_test)
            elif str(image_id).zfill(12) in val_list:
                anno_val['area'] = area
                anno_val['iscrowd'] = iscrowd
                anno_val['image_id'] = image_id
                anno_val['bbox'] = bbox
                anno_val['category_id'] = category_id
                anno_val['id'] = id
                annos_val.append(anno_val)
            else:
                anno_train['area'] = area
                anno_train['iscrowd'] = iscrowd
                anno_train['image_id'] = image_id
                anno_train['bbox'] = bbox
                anno_train['category_id'] = category_id
                anno_train['id'] = id
                annos_train.append(anno_train)

        categories = root['categories']

    context_train['annotations'] = annos_train
    context_test['annotations'] = annos_test
    context_val['annotations'] = annos_val

    context_train['images'] = imgs_train
    context_test['images'] = imgs_test
    context_val['images'] = imgs_val

    context_train['categories'] = categories
    context_test['categories'] = categories
    context_val['categories'] = categories

    json.dump(context_train, open(save_train, 'w'), indent=1)
    json.dump(context_test, open(save_test, 'w'), indent=1)
    json.dump(context_val, open(save_val, 'w'), indent=1)
    print('json written down.')


if __name__ == '__main__':
    # parse_anno()
    # split_dataset(image_root)
    get_annotations(anno_root)

import os
import random
import numpy as np
import json

from tqdm import tqdm

dataset_dir = 'E:/300W_LP_Out'

train_json = 'data_train'
val_json = 'data_test'
img_size = 160
eye_close_thres = 0.02
mouth_close_thres = 0.02
big_mouth_open_thres = 0.08
ratio = 0.95


def scan_dataset(path, shuffle: bool = False):
    files_list = []

    def inner(dir_path):
        fs = os.scandir(dir_path)
        for f in fs:
            name: str = f.name
            if f.is_file():
                if name.endswith(('.jpg', '.png', '.jpeg')):
                    files_list.append(f.path)
            elif f.is_dir():
                inner(f.path)

    inner(path)
    if shuffle:
        random.shuffle(files_list)
    return files_list


def split(datas: list, rate: float = 0.95):
    length = int(rate * len(datas))
    train = datas[:length]
    test = datas[length:]
    return train, test


def generate_json(name_list: list, filename: str):
    json_list = []
    for pic in tqdm(name_list):
        pic: str = pic
        one_image_ann = dict()
        one_image_ann['image_path'] = pic
        pts = pic.rsplit('.', 1)[0] + '.pts'
        try:
            tmp = []
            with open(pts) as p_f:
                labels = p_f.readlines()[3:-1]
            for _one_p in labels:
                xy = _one_p.rstrip().split(' ')
                tmp.append([float(xy[0]), float(xy[1])])
            one_image_ann['keypoints'] = tmp
            label = np.array(tmp).reshape((-1, 2))  # 转换为ndarry格式数据
            bbox = [float(np.min(label[:, 0])), float(np.min(label[:, 1])), float(np.max(label[:, 0])),
                    float(np.max(label[:, 1]))]
            one_image_ann['bbox'] = bbox
            json_list.append(one_image_ann)
        except IOError:
            raise Exception("未正确打开文件")
    if not filename.endswith(".json"):
        filename += '.json'
    with open(filename, 'w') as json_file:
        json.dump(json_list, json_file, indent=2)


if __name__ == '__main__':
    data = scan_dataset(dataset_dir, True)
    train_list, val_list = split(data)
    generate_json(train_list, train_json)
    generate_json(val_list, val_json)

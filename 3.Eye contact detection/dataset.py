from torch.utils.data.dataset import Dataset
from util import random_crop, random_rotate, pixel_jitter
import random
import numpy as np
import json
import os
import cv2


def scan_dir(dirpath: str, filetypes: tuple, shuffle: bool = False):
    files_list = []

    def inner(dir_path):
        files = os.scandir(dir_path)
        for item in files:
            name: str = item.name
            if item.is_file():
                if name.endswith(filetypes):
                    files_list.append(item.path)
            elif item.is_dir():
                inner(item.path)

    inner(dirpath)
    if shuffle:
        random.shuffle(files_list)
    return files_list


def split_test_dataset(testdata: str):
    f = open(testdata, 'r')
    tmp = []
    tmp1 = []
    data = json.load(f)
    for d in data:
        lpath: str = d["left"]
        if lpath.find("300wdataset") == -1:
            tmp.append(d)
        else:
            tmp1.append(d)
    f1 = open("testno300w_a.json", "w")
    f2 = open("test300w_a.json", "w")

    json.dump(tmp1, f2, indent=2)
    json.dump(tmp, f1, indent=2)
    f2.close()
    f1.close()


def split_json_dataset(jsonfile: str, scale_for_train: float):
    f = open(jsonfile, "r")
    alldata = json.load(f)
    tl = len(alldata)
    random.shuffle(alldata)
    idx = int(scale_for_train * tl)
    print("{} for trian; {} for test".format(idx, tl - idx))
    train = alldata[:idx]
    test = alldata[idx:]
    ftrain = open("train_data_a.json", 'w+')
    ftest = open("test_data_a.json", 'w+')
    json.dump(train, ftrain, indent=2)
    json.dump(test, ftest, indent=2)
    ftrain.close()
    ftest.close()


def fusion_dataset():
    d1 = open("gdataset_a.json", "r")
    d2 = open("300wd_a.json", "r")
    d1l: list = json.load(d1)
    d2l: list = json.load(d2)
    d1l.extend(d2l)
    random.shuffle(d1l)
    alldata = open("alldata_a.json", "w")
    json.dump(d1l, alldata, indent=2)
    alldata.close()


class DataSet(Dataset):
    def __init__(self, json_file: str, is_trian: bool = False):
        fp = open(json_file, "r")
        self.data = json.load(fp)
        self.length = len(self.data)
        fp.close()
        self.train_flag = is_trian

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d: dict = self.data[index]
        limg = cv2.imread(d["left"])
        rimg = cv2.imread(d["right"])
        # 下采样

        h, w, _ = limg.shape
        if h > 60 and w > 100:
            if self.train_flag:
                resize_m = random.choice([cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_LANCZOS4])
            else:
                resize_m = cv2.INTER_LINEAR
            limg = cv2.resize(limg, (100, 60), interpolation=resize_m)
            rimg = cv2.resize(rimg, (100, 60), interpolation=resize_m)

        if self.train_flag:
            c = random.choice([0, 1, 2, 3])
            if c == 0:
                limg = random_crop(limg, 0.06)
            if c == 1:
                rimg = random_crop(rimg, 0.06)

            c1 = random.choice([0, 1, 2, 3])
            angle = random.uniform(-8, 8)
            if c1 == 0:
                limg = random_rotate(limg, angle)
            if c1 == 1:
                rimg = random_rotate(rimg, angle)

            c2 = random.choice([0, 1, 2, 3])
            scale = random.randint(1, 6)
            if c2 == 0:
                limg = pixel_jitter(limg, scale)
            if c2 == 1:
                rimg = pixel_jitter(rimg, scale)

        limg = np.transpose(limg, (2, 0, 1)) / 255 * 0.99 + 0.01
        rimg = np.transpose(rimg, (2, 0, 1)) / 255 * 0.99 + 0.01
        limg = limg.astype(np.float32)
        rimg = rimg.astype(np.float32)

        eyeon: bool = d['label']
        if eyeon:
            label = np.array([0.1, 99.9])
        else:
            label = np.array([99.9, 0.1])
        label = label.astype(np.float32)
        if d.get("pitch"):
            pitch = d["pitch"]
            yaw = d["yaw"]
            roll = d["roll"]
            angles = np.array([pitch, yaw, roll])
            angles = angles.astype(np.float32)
            return limg, rimg, label, angles
        else:
            return limg, rimg, label


if __name__ == '__main__':
    # get_json('E:/Columbia Gaze Data Set', "E:/myproject/EyeContact/dataset", "dataset.json")
    # split_json_dataset("alldata_a.json", 0.85)
    split_test_dataset("test_data_a.json")
    # fusion_dataset()
# gd = GDataSet("dataset.json", True)
# print(gd[4])
# cimg = ((gd[1][1] - 0.01) / 0.99 * 255).astype(np.uint8)
# print(cimg.shape)
# cimg = np.transpose(cimg, (1, 2, 0))
# cv2.imshow("11", cimg)
# cv2.waitKey(0)

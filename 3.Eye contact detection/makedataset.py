import cv2
import os
import csv
import numpy as np
import math
import json
from os import path
from headpose import get_head_pose


def point_dis(p1x, p1y, p2x, p2y) -> float:
    dis = (p1x - p2x) ** 2 + (p1y - p2y) ** 2
    return math.sqrt(dis)


def get_box(img, shape, base, h, w):
    width = point_dis(shape[base][0], shape[base][1], shape[base + 3][0], shape[base + 3][1])
    b = np.arccosh(width / abs(shape[base][0] - shape[base + 3][0]))
    height = max(point_dis(shape[base + 1][0], shape[base + 1][1], shape[base + 5][0], shape[base + 5][1]),
                 point_dis(shape[base + 2][0], shape[base + 2][1], shape[base + 4][0], shape[base + 4][1]))

    # height = max(abs(shape[base + 1][1] - shape[base + 5][1]), abs(shape[base + 2][1] - shape[base + 4][1]))
    center = ((shape[base + 1][0] + shape[base + 2][0] + shape[base + 5][0] + shape[base + 4][0]) / 4,
              (shape[base + 1][1] + shape[base + 2][1] + shape[base + 5][1] + shape[base + 4][1]) / 4)
    th = (width * h / w) * 1.2 if height < width * h / w else height * 1.2
    # b = np.arccosh(width / abs(shape[base][0] - shape[base + 3][0]))
    lt = (int(center[0] - width * 0.7), int(center[1] - th / 2))
    rb = (int(center[0] + width * 0.7), int(center[1] + th / 2))
    rows, cols, channels = img.shape
    rotate = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), -np.degrees(b), 1)
    imgr = cv2.warpAffine(img.copy(), rotate, (cols, rows), borderValue=(255, 255, 255), flags=cv2.INTER_LANCZOS4)
    croped = imgr[lt[1]:rb[1], lt[0]:rb[0]]
    if h and w:
        return cv2.resize(croped, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return croped


def get_all_pts():
    d = {}
    all_pts_file = os.scandir("E:/dataset/dataset")
    for file in all_pts_file:
        if file.name.endswith("pts"):
            d[file.name.split('.')[-2]] = file.path
    return d


ptsdict = get_all_pts()


def all_image_from_300w(dir_name: str, writeimg: bool = False):
    f = open("E:/dataset/annotation.csv")
    reader = csv.reader(f)
    index = 0
    jdname = open("300wd.json", 'w')
    jdata = []
    for imag, status in reader:
        fname = imag.split('.')[-2]
        ptsfile = ptsdict[fname]
        tmp = []
        f = open(ptsfile)
        ptsdata = f.readlines()[3:-1]
        for _one_p in ptsdata:
            xy = _one_p.rstrip().split(' ')
            tmp.append([float(xy[0]), float(xy[1])])
        shape = np.array(tmp).reshape((-1, 2))
        f.close()
        if status == "0":
            label = False
        else:
            label = True
        ln = fname + "_l.jpg"
        rn = fname + "_r.jpg"
        fln = fname + "_flip" + "_l.jpg"
        frn = fname + "_flip" + "_r.jpg"
        fpath = path.join("E:/dataset/filter", imag)
        original_image = cv2.imread(fpath)
        if writeimg:
            corped_img_left = get_box(original_image, shape, 36, 60, 100)
            corped_img_right = get_box(original_image, shape, 42, 60, 100)
            flip_left = cv2.flip(corped_img_left, 1)
            flip_right = cv2.flip(corped_img_right, 1)
            os.chdir(dir_name)
            cv2.imwrite(ln, corped_img_left)
            cv2.imwrite(rn, corped_img_right)
            cv2.imwrite(fln, flip_left)
            cv2.imwrite(frn, flip_right)
        _, pose = get_head_pose(shape, original_image)
        pose = pose.flatten()

        d = {"left": path.join(dir_name, ln),
             "right": path.join(dir_name, rn),
             "label": label,
             "pitch": pose[0],
             "yaw": pose[1],
             "roll": pose[2]}
        df = {"left": path.join(dir_name, fln),
              "right": path.join(dir_name, frn),
              "label": label,
              "pitch": -pose[0],
              "yaw": pose[1],
              "roll": pose[2]}
        jdata.append(d)
        jdata.append(df)
        index += 1
        print(index)
    json.dump(jdata, jdname, indent=2)
    jdname.close()


def dataset_stat():
    f = open('annotation.json')
    js: dict = json.load(f)
    postive, negitive = 0, 0
    for v in js.values():
        if v == 1:
            postive += 1
        elif v == 0:
            negitive += 1
    return postive, negitive


if __name__ == '__main__':
    all_image_from_300w("E:/myproject/EyeContact/300wdata1",True)

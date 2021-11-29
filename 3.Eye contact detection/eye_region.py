import dlib
import cv2
import numpy as np
import random
import json
from dataset import scan_dir
from os import path
from makedataset import get_box
from headpose import get_head_pose

detector = dlib.get_frontal_face_detector()
face_shape = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

left = 36  # keypoint indices for left eye
right = 42  # keypoint indices for right eye


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def judge(filename: str):
    eles = filename.strip(".jpg").split("_")
    if eles[-1] == "0H" and eles[-2] == "0V":
        return True
    return False


def make_dataset(base_dir: str, h, w, jfname: str, writeimg: bool = False):
    imgpaths = scan_dir('E:/Columbia Gaze Data Set', ("jpg",))
    tmpdata = []
    leftdata = []
    jf = open(jfname, "w")
    unrec = open("left.json", "w")
    total = len(imgpaths)
    for imgidx in range(total):
        imgpath = imgpaths[imgidx]
        img = cv2.imread(imgpath)
        face = detector(img, 0)
        if len(face) > 0:
            shapes = face_shape(img, face[0])
            shape = shape_to_np(shapes)
            filename = path.basename(imgpath)[:-4]
            lfn = base_dir + filename + "_l.jpg"
            rfn = base_dir + filename + "_r.jpg"
            _, pose = get_head_pose(shape, img)
            if writeimg:
                le = get_box(img, shape, 36, h, w)
                re = get_box(img, shape, 42, h, w)
                cv2.imwrite(lfn, le)
                cv2.imwrite(rfn, re)
            pose = pose.flatten()
            label = judge(imgpath)
            d = {"left": path.join(base_dir, lfn),
                 "right": path.join(base_dir, rfn),
                 "label": label,
                 "pitch": pose[0],
                 "yaw": pose[1],
                 "roll": pose[2]}
            if label:
                traget = [d] * 20
                tmpdata.extend(traget)
            else:
                tmpdata.append(d)
        else:
            leftdata.append(imgpath)
        print("{}/{}".format(imgidx, total))
    random.shuffle(tmpdata)
    json.dump(tmpdata, jf, indent=2)
    json.dump(leftdata, unrec, indent=2)


# if __name__ == '__main__':
    # make_dataset("E:/myproject/EyeContact/dataset/", 120, 200, "gdata.json")
    # exist = scan_dir('E:/myproject/EyeContact/dataset/', ("jpg",))
    # exist = list(map(path.basename, map(lambda x: x[:-6] + x[-4:], exist)))
    # imgs_c = list(map(path.basename, imgs))
    # res = set(imgs_c) - set(exist)
    # leftimg = []
    # for img in imgs:
    #     if path.basename(img) in res:
    #         leftimg.append(path.abspath(img))
    # f = open("left.json", "w")
    # json.dump(leftimg, f)
    # f.close()
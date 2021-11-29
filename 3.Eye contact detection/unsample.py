import cv2
import numpy as np
import os
from eye_region import judge
from dataset import scan_dir
from os import path
import json


def down_sample(img: np.ndarray, scale: float, method):
    h, w, _ = img.shape
    nh, nw = int(h * scale), int(w * scale)
    simg = cv2.resize(cv2.resize(img, (nw, nh), interpolation=method),
                      (w, h), interpolation=method)
    return simg


ratio = [0.70, 0.5, 0.3, 0.15]
ms = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

if __name__ == '__main__':
    # data = json.load(open("eyeon.json", "r"))
    # nr = []
    # basep = 'E:/myproject/EyeContact/dataset/'
    # for d in data:
    #     dc: dict = d.copy()
    #     for r in ratio:
    #         for m in ms:
    #             dc["left"] = path.join(basep, "({}-{}){}".format(r, m, path.basename(d["left"])))
    #             dc["right"] = path.join(basep, "({}-{}){}".format(r, m, path.basename(d["right"])))
    #             nr.append(dc.copy())
    # print(nr)
    # print(len(nr))
    # json.dump(nr, open("res.json", "w"), indent=2)

    # imgs = scan_dir("E:/Columbia Gaze Data Set", ("jpg",))
    # eyeson = list(map(path.basename, list(filter(judge, imgs))))
    # links = []
    # basepath = "E:/myproject/EyeContact/dataset"
    # for imglink in eyeson:
    #     lname = imglink[:-4] + "_l" + imglink[-4:]
    #     rname = imglink[:-4] + "_r" + imglink[-4:]
    #     links.append(path.join(basepath, lname))
    #     links.append(path.join(basepath, rname))
    # os.chdir("E:/myproject/EyeContact/downsamples")
    # idx = 0
    # print(links)
    # print(len(links))
    # for eyesonimg in links:
    #     fnmae = path.basename(eyesonimg)
    #     oimg = cv2.imread(eyesonimg)
    #     for r in ratio:
    #         for m in ms:
    #             s = down_sample(oimg, r, m)
    #             imgname = "({}-{}){}".format(r, m, fnmae)
    #             cv2.imwrite(imgname, s)
    #             idx += 1
    #             print(idx)
    eyeon = json.load(open("eyeon.json", "r"))
    eyesonj = json.load(open("res.json", "r"))
    ori = json.load(open("gdataset_a.json", "r"))
    a = list(filter(lambda x: not x["label"], ori))
    a.extend(eyeon)
    a.extend(eyesonj)
    json.dump(a, open("gall_a.json", "w"), indent=2)

import dlib
import cv2
import numpy as np
import os

detector = dlib.get_frontal_face_detector()
face_shape = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

left = [36, 37, 38, 39, 40, 41]  # keypoint indices for left eye
right = [42, 43, 44, 45, 46, 47]  # keypoint indices for right eye


# camera = cv2.VideoCapture(0)
# while camera.isOpened():
#     frame, status = camera.read()
#
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def get_box(img, shape, base, h, w):
    width = abs(shape[base][0] - shape[base + 3][0])
    height = max(abs(shape[base + 1][1] - shape[base + 5][1]), abs(shape[base + 2][1] - shape[base + 4][1]))
    center = (shape[base][0] + width / 2, shape[base + 1][1] + height / 2)
    th = (width * h / w) * 1.2 if height < width * h / w else height * 1.2
    b = np.arctan(abs(shape[base][1] - shape[base + 3][1]) / width)
    lt = (int(center[0] - width * 0.7), int(center[1] - th / 2))
    rb = (int(center[0] + width * 0.7), int(center[1] + th / 2))
    rows, cols, channels = img.shape
    rotate = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), -np.degrees(b), 1)
    imgr = cv2.warpAffine(img.copy(), rotate, (cols, rows), borderValue=(255, 255, 255), flags=cv2.INTER_LANCZOS4)
    croped = imgr[lt[1]:rb[1], lt[0]:rb[0]]
    # if h and w:
    #     return cv2.resize(croped, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return croped


def get_two(path: str, h, w):
    output = "E:/myproject/EyeContact/handled/"
    img = cv2.imread(path)
    face = detector(img, 0)
    if len(face) > 0:
        shapes = face_shape(img, face[0])
        shape = shape_to_np(shapes)
        le = get_box(img, shape, 36, h, w)
        re = get_box(img, shape, 42, h, w)
        filename = path.split("/")[-1][:-4]
        lfn = output + filename + "l.jpg"
        rfn = output + filename + "r.jpg"
        cv2.imwrite(lfn, le)
        cv2.imwrite(rfn, re)


# l1, l2 = get_two(None, 60, 100)
# cv2.imwrite("le.jpg", l1)
# cv2.imwrite("re.jpg", l2)


def handle():
    fs = os.scandir("C:/Users/Xuan/Desktop/dataset/filter")
    for f in fs:
        p: str = f.path
        path = p.replace("\\", "/")
        get_two(path, 90, 150)


if __name__ == '__main__':
    handle()

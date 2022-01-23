import torch
import cv2
import numpy as np
from typing import Tuple
from face_onnx.detector import Detector as FaceDetector
from network import SubCNNA, NetA
from torch import load
from detector import Detector
from makedataset import get_box

face_detector = FaceDetector()
lmk_detector = Detector()

model = NetA(SubCNNA)
model.load_state_dict(load("eye_contact.chk"))
model.cpu()


def predict(frame) -> Tuple[bool, str]:
    boxes, _ = face_detector.detect(frame)
    if len(boxes) == 0:
        return False, "未检测到人脸"
    facebox = boxes[0].astype(np.int32)
    lmk, pose = lmk_detector.detect(frame, facebox)
    # pose [Pitch,Yaw,Roll]
    if abs(pose[0]) >= 15 or abs(pose[1]) >= 30:
        return False, "头部偏转角度过大"
    lefteye = get_box(frame, lmk, 36, 60, 100)
    righteye = get_box(frame, lmk, 36, 60, 100)
    if not predict_eye_contact(lefteye, righteye, pose):
        return False, "视线未接触"
    return True, "关注状态"


def predict_eye_contact(leftimg, rightimg, angles) -> bool:
    li: np.ndarray = np.transpose(leftimg, (2, 0, 1)) / 255 * 0.99 + 0.01
    ri: np.ndarray = np.transpose(rightimg, (2, 0, 1)) / 255 * 0.99 + 0.01
    li = li.reshape((-1, 3, 60, 100))
    ri = ri.reshape((-1, 3, 60, 100))
    limg = torch.tensor(li, dtype=torch.float32)
    rimg = torch.tensor(ri, dtype=torch.float32)
    angle = torch.tensor(angles, dtype=torch.float32)
    angle = angle.reshape([-1, 1, 3])
    predres = model(limg, rimg, angle)
    result = predres.tolist()[0]
    return result[0] != max(result)


def demo(width: int = 640, height: int = 480):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = predict(frame)
        cv2.putText(frame, str(res), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 1, 1)

        cv2.imshow("Test", frame)
        if cv2.waitKey(27) == ord("q"):
            break


if __name__ == '__main__':
    demo()

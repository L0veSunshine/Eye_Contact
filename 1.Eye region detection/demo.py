import cv2
import numpy as np
from face_onnx.detector import Detector as FaceDetector
from detector import Detector

face_detector = FaceDetector()
lmk_detector = Detector()
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    bboxes, _ = face_detector.detect(frame)
    if len(bboxes) != 0:
        bbox = bboxes[0]
        bbox = bbox.astype(np.int64)
        lmks, PRY_3d = lmk_detector.detect(frame, bbox)
        lmks = lmks.astype(np.int64)
        frame = cv2.rectangle(frame, tuple(bbox[0:2]), tuple(bbox[2:4]), (0, 0, 255), 1, 1)
        for point in lmks:
            frame = cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1, 1)
        # pitch 物体绕X轴旋转
        frame = cv2.putText(frame, "Pitch: {:.4f}".format(PRY_3d[0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 1, 1)
        # yaw 物体绕Y轴旋转
        frame = cv2.putText(frame, "Yaw: {:.4f}".format(PRY_3d[1]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 1, 1)
        # roll 物体绕Z轴旋转
        frame = cv2.putText(frame, "Roll: {:.4f}".format(PRY_3d[2]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 1, 1)
    cv2.imshow("Test", frame)
    if cv2.waitKey(27) == ord("q"):
        break

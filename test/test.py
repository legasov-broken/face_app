import matplotlib.pyplot as plt
from retinaface import RetinaFace
import cv2
import torch

img_path = "/home/minelove/Face-Detection/test/IMG_1439.JPEG"
faces = RetinaFace.detect_faces(img_path)
img = cv2.imread(img_path)

for face_name, face in faces.items():
    identity = face

    facial_area = identity['facial_area']
    img = cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (0, 255, 0), 15)
    
cv2.imwrite("out.JPG", img)
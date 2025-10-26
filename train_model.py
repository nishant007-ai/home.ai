import cv2
import numpy as np
import os

data_path = 'data/'
labels, faces = [], []
label_map = {}
i = 0

for person in os.listdir(data_path):
    person_path = os.path.join(data_path, person)
    label_map[i] = person
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(i)
    i += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("models/face_model.yml")

print("✅ Model trained and saved!")

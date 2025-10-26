import cv2
import face_recognition
import os
from datetime import datetime
from attendance_marker import mark_attendance

path = 'data/known_faces'
images = []
classNames = []

for filename in os.listdir(path):
    img = cv2.imread(f'{path}/{filename}')
    images.append(img)
    classNames.append(os.path.splitext(filename)[0])

def encode_faces(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodelist.append(face_recognition.face_encodings(img)[0])
    return encodelist

known_encodings = encode_faces(images)

def recognize_faces():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break
        small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

        faces_cur_frame = face_recognition.face_locations(small_img)
        encodings_cur_frame = face_recognition.face_encodings(small_img, faces_cur_frame)

        for encode_face, face_loc in zip(encodings_cur_frame, faces_cur_frame):
            matches = face_recognition.compare_faces(known_encodings, encode_face)
            face_dist = face_recognition.face_distance(known_encodings, encode_face)
            best_match = min(range(len(face_dist)), key=face_dist.__getitem__)
            if matches[best_match]:
                name = classNames[best_match]
                mark_attendance(name)
        break
    cap.release()

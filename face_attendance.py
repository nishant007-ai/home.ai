import face_recognition
import os
import cv2
import csv
from datetime import datetime

# Directory of known faces
known_faces_dir = "data/known_faces"
captured_image_path = "captured_photo.jpg"
attendance_file = "attendance.csv"

print("📥 Loading known faces...")
known_encodings = []
known_names = []

# Load all known face encodings
for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg", ".png")):
        filepath = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_names.append(name)
            print(f"✅ Face loaded for: {name}")
        else:
            print(f"⚠️ No face found in {filename}")

print(f"Total known faces: {len(known_names)}")

# Load and encode the captured photo
print("🖼️ Reading captured image...")
captured_image = face_recognition.load_image_file(captured_image_path)
captured_encodings = face_recognition.face_encodings(captured_image)

if not captured_encodings:
    print("❌ No face found in captured photo.")
    exit()

# Compare with known encodings
matched_name = "Unknown"
for captured_encoding in captured_encodings:
    matches = face_recognition.compare_faces(known_encodings, captured_encoding)
    face_distances = face_recognition.face_distance(known_encodings, captured_encoding)

    if True in matches:
        best_match_index = face_distances.argmin()
        matched_name = known_names[best_match_index]
        print(f"🎯 Match found: {matched_name}")
        break

# Mark attendance
if matched_name != "Unknown":
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    with open(attendance_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([matched_name, timestamp])
    
    print(f"📝 Attendance marked for {matched_name} at {timestamp}")
else:
    print("❌ No matching face found.")

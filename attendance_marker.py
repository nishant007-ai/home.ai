import face_recognition
import os
import cv2
import csv
from datetime import datetime

print("📥 Loading known faces...")

known_faces_dir = "data/known_faces"
known_encodings = []
known_names = []

# Load known face images
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        print(f"👉 Processing: {filename}")
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)

        if encoding:
            known_encodings.append(encoding[0])
            known_names.append(os.path.splitext(filename)[0])
        else:
            print(f"⚠️ No face found in {filename}")

print("✅ Known faces loaded:", known_names)

# Load captured photo
print("🖼️ Reading captured image...")
captured_image = face_recognition.load_image_file("captured_photo.jpg")
captured_encodings = face_recognition.face_encodings(captured_image)

if not captured_encodings:
    print("❌ No face found in captured_photo.jpg")
    exit()

matched_name = "Unknown"

# Compare with known faces
for captured_encoding in captured_encodings:
    matches = face_recognition.compare_faces(known_encodings, captured_encoding)
    face_distances = face_recognition.face_distance(known_encodings, captured_encoding)

    if True in matches:
        best_match_index = face_distances.argmin()
        matched_name = known_names[best_match_index]
        print(f"✅ Match found: {matched_name}")
        break

if matched_name != "Unknown":
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

    # Save to CSV
    with open("attendance.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([matched_name, dt_string])
    print(f"📝 Attendance marked for {matched_name} at {dt_string}")
else:
    print("❌ No match found in known faces.")

from flask import Flask, render_template, request
import face_recognition
import os
import datetime
import csv

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
KNOWN_FOLDER = 'known_faces'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load known faces
known_encodings = []
known_names = []

print("📥 Loading known faces...")
for filename in os.listdir(KNOWN_FOLDER):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(KNOWN_FOLDER, filename)
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_encodings.append(encoding[0])
            name = os.path.splitext(filename)[0]
            known_names.append(name)
            print(f"✅ Face loaded for: {name}")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            unknown_image = face_recognition.load_image_file(filepath)
            unknown_encodings = face_recognition.face_encodings(unknown_image)

            if unknown_encodings:
                match = face_recognition.compare_faces(known_encodings, unknown_encodings[0])
                if True in match:
                    matched_name = known_names[match.index(True)]
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Mark attendance
                    with open('attendance.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([matched_name, now])

                    result = f"🎯 Match found: {matched_name}<br>📝 Attendance marked at {now}"
                else:
                    result = "❌ No match found."
            else:
                result = "⚠️ No face detected in the uploaded image."

            os.remove(filepath)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    print("🚀 Server running at http://127.0.0.1:5000")
    app.run(debug=True)

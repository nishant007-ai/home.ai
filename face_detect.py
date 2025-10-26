import cv2

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw green rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame with rectangles
    cv2.imshow("Live Feed - Press SPACE to capture", frame)

    key = cv2.waitKey(1)
    if key == 32:  # SPACE bar to capture
        # Save the frame with rectangles
        cv2.imwrite("captured_photo.jpg", frame)
        print("✅ Image with face(s) saved as captured_photo.jpg")
        break
    elif key == 27:  # ESC to quit
        print("❌ Capture cancelled")
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

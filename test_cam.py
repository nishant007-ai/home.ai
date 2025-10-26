import cv2

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Use this if you're on macOS

if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

ret, frame = cap.read()
if ret:
    cv2.imshow("Captured Image", frame)

    # Save the image
    cv2.imwrite("captured_photo.jpg", frame)
    print("✅ Image saved as captured_photo.jpg")

    cv2.waitKey(0)
else:
    print("❌ Failed to capture image")

cap.release()
cv2.destroyAllWindows()

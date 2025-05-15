import cv2
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.saving import register_keras_serializable

# Register Sequential model to avoid deserialization issue
@register_keras_serializable()
class MySequential(Sequential):
    pass

# Load model architecture
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()

# Load model with registered class
model = model_from_json(model_json, custom_objects={"Sequential": MySequential})
model.load_weights("emotiondetector.h5")  # Load model weights

# Load Haarcascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels
labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Function to preprocess input image
def extract_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1)  # Reshape for model input
    return feature / 255.0  # Normalize pixel values

# Start webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Extract face ROI
        face = cv2.resize(face, (48, 48))  # Resize for model
        img = extract_features(face)

        # Predict emotion
        pred = model.predict(img)
        emotion_label = labels[pred.argmax()]

        # Draw bounding box & emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Emotion Detection", frame)  # Show frame

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()

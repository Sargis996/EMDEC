from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

# Load Haar Cascade Classifier for face detection

face_classifier = cv2.CascadeClassifier(r'C:\Users\Admin\Desktop\PythonProject\EmotionDetectionCNN\haarcascade_frontalface_default.xml')

# Load pre-trained Emotion Detection CNN model

classifier =load_model(r'C:\Users\Admin\Desktop\PythonProject\EmotionDetectionCNN\model.h5')

# Emotion labels for classification

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Open a connection to the camera (Camera 0)

cap = cv2.VideoCapture(0)



while True:
        # Read a frame from the camera

    _, frame = cap.read()
    labels = []
     # Convert the frame to grayscale for face detection

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     # Detect faces in the frame
    faces = face_classifier.detectMultiScale(gray)
    # Iterate through detected faces
    for (x,y,w,h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        # Resize the face region for emotion prediction
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


# Check if the face region is not empty
        if np.sum([roi_gray])!=0:
            # Normalize and preprocess the face for emotion prediction
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
            # Make emotion prediction using the loaded model
            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            # Display the predicted emotion label on the frame
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            # Display "No Faces" if no face is detected
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
   # Show the frame with emotion detection
    cv2.imshow('Emotion Detector',frame)
  # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

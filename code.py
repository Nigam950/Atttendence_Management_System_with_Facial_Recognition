import cv2
import os

def capture_faces(person_name):
    if not os.path.exists(f"dataset/{person_name}"):
        os.makedirs(f"dataset/{person_name}")
    
    cap = cv2.VideoCapture(0)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            face = gray[y:y + h, x:x + w]
            cv2.imwrite(f"dataset/{person_name}/{person_name}_{count}.jpg", face)
            count += 1

        cv2.imshow('Capturing Faces', frame)

        if count >= 100:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

capture_faces("John_Doe")

import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

def train_model():
    faces = []
    labels = []
    label_names = []

    for person_name in os.listdir("dataset"):
        for image_name in os.listdir(f"dataset/{person_name}"):
            img = cv2.imread(f"dataset/{person_name}/{image_name}", cv2.IMREAD_GRAYSCALE)
            face = cv2.resize(img, (100, 100)).flatten()  
            faces.append(face)
            labels.append(person_name)
        
    faces = np.array(faces)
    labels = np.array(labels)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    
    pca = PCA(n_components=50)
    faces = pca.fit_transform(faces)

    svm = SVC(kernel='linear', probability=True)
    svm.fit(faces, labels)

    with open("face_model.pkl", "wb") as f:
        pickle.dump((pca, le, svm), f)

train_model()


import cv2
import numpy as np
import pandas as pd
import pickle
import datetime

with open("face_model.pkl", "rb") as f:
    pca, le, svm = pickle.load(f)


cap = cv2.VideoCapture(0)

attendance_log = pd.DataFrame(columns=["Name", "Time"])

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (100, 100)).flatten()  # Flatten the face image

    
        face_pca = pca.transform([face_resized])

        
        prediction = svm.predict(face_pca)
        person_name = le.inverse_transform(prediction)[0]

       
        if person_name not in attendance_log["Name"].values:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            attendance_log = attendance_log.append({"Name": person_name, "Time": current_time}, ignore_index=True)
            print(f"Attendance marked for {person_name} at {current_time}")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Attendance System", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

attendance_log.to_csv("attendance_log.csv", index=False)
cap.release()
cv2.destroyAllWindows()


print("Attendance marked:")
print(attendance_log)
n
import matplotlib.pyplot as plt

attendance = pd.read_csv("attendance_log.csv")
n
attendance_count = attendance["Name"].value_counts()

attendance_count.plot(kind='bar', color='skyblue', figsize=(10, 6))
plt.title("Attendance Distribution")
plt.xlabel("Names")
plt.ylabel("Attendance Count")
plt.xticks(rotation=45)
plt.show()

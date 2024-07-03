import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Directory containing the images
path = 'images'
images = []
classNames = []

# Load images and their corresponding class names
for person_dir in os.listdir(path):
    person_path = os.path.join(path, person_dir)
    if os.path.isdir(person_path):
        for image_file in os.listdir(person_path):
            img_path = os.path.join(person_path, image_file)
            curImg = cv2.imread(img_path)
            if curImg is not None:
                images.append(curImg)
                classNames.append(person_dir)
            else:
                print(f"Failed to load image: {img_path}")

print(f'Class names: {classNames}')

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Mark_Attendance.csv', 'a+') as f:
        f.seek(0)
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            dateString = now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{dtString},{dateString}')

# Find encodings of known faces
try:
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')
except Exception as e:
    print(f'Error encoding images: {e}')
    exit()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break
    
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        
        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            markAttendance(name)
            print(f'Recognized: {name}')
            color = (0, 255, 0)  # Green for recognized
        else:
            name = 'Unknown'
            color = (0, 0, 255)  # Red for unknown
            print(f'Unknown face detected.')
        
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Webcam", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
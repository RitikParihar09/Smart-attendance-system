import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'StudentsImages'
images = []
classNames = []
myList = os.listdir(path)


path = 'StudentsImages'
if os.path.exists(path):
    print("Folder found:", path)
else:
    print("Folder not found:", path)


for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is None:
        print(f"Warning: Unable to load image {cl}")
        continue
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print("Class Names:", classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            print("Warning: No face found in one of the images.")
    return encodeList

encodeListKnown = findEncodings(images)
if not encodeListKnown:
    print("Error: No face encodings found. Ensure the 'StudentsImages' folder contains valid images.")
    exit()
print("Encoding Complete. Encodings found:", len(encodeListKnown))


marked_names = set()
def markAttendance(name):
    global marked_names  
    
    if name not in marked_names:
        marked_names.add(name)
        
        now = datetime.now()
        dateString = now.strftime('%Y-%m-%d') 
        timeString = now.strftime('%H:%M:%S') 
        
        
        with open('Attendance.csv', 'a') as f:
            f.writelines(f'{name},{dateString},{timeString}\n')
        print(f"Attendance marked for {name}.")
    else:
        print(f"{name} already marked for this session.")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access webcam.")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to read frame from webcam.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                font_scale = 1
                font_thickness = 2
                text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_COMPLEX, font_scale, font_thickness)[0]
                text_width, text_height = text_size[0], text_size[1]

                cv2.rectangle(img, (x1, y2 + 10), (x1 + text_width + 20, y2 + text_height + 30), (0, 255, 0), cv2.FILLED)

                cv2.putText(img, name, (x1 + 10, y2 + text_height + 25), cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255), font_thickness)

                markAttendance(name)

        else:
            print("Warning: No matching faces found in the current frame.")

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

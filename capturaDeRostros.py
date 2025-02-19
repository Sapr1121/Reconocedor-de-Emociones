import cv2
import os
import imutils

#emotionName = 'Enojo'
#emotionName = 'Tristeza'
#emotionName = 'Felicidad'
#emotionName = 'Sorpresa'
emotionName = 'Normal'

dataPath = 'C:/Users/sapr2/Desktop/Personal/ReconocimientoEmociones/data'
emotionsPath = dataPath + '/' + emotionName

if not os.path.exists(emotionsPath):
    os.mkdir(emotionsPath)
    print(f'Carpeta creada: {emotionsPath}')
    
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()  # Asegúrate de tener paréntesis aquí

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(emotionsPath + '/rostro_{}.jpg'.format(count), rostro)
        count += 1

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 200:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import os
import numpy as np
import time

def obtenerModelo(method, facesData, labels):
    if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
    if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
    if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

    #Entrenar el reconocedor
    print('entrenando ('+method+')...')
    inicio = time.time()
    emotion_recognizer.train(facesData, np.array(labels))
    tiempoDeEntrenamiento = time.time()-inicio
    print('Tiempo de entrenamiento ('+method+'): ',tiempoDeEntrenamiento)
    emotion_recognizer.write('modelo'+method+'.xml')

dataPath = 'C:/Users/sapr2/Desktop/Personal/ReconocimientoEmociones/data'
emotionList = os.listdir(dataPath)
print("Lista de personas: ", emotionList)


labels = []
facesData = []
label = 0

for nameDir in emotionList:
    emotionPath = dataPath + '/' + nameDir
    #print('Leyendo Imagenes')

    for fileName in os.listdir(emotionPath):
        #print('Rostros: ', nameDir + '/' + fileName,0)
        labels.append(label)
        facesData.append(cv2.imread(emotionPath + '/' + fileName,0))
    label+=1

obtenerModelo('EigenFaces', facesData, labels)
obtenerModelo('FisherFaces', facesData, labels)
obtenerModelo('LBPH', facesData, labels)






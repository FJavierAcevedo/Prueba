# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:52:25 2022

@author: Cristian
"""
# Este script realiza la detección de rostros con mediapipe, obteniendo el valor de lso pixeles térmicos en un flujo de video

import mediapipe as mp
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import glob

        
def draw_termica(x,y,frame0):
# dibujar un circulo por cada id
    color=[(0,255,0),(255,0,0),(0,0,255)]
    #color=(255,0,0)
    A1=np.array([[x],[y],[1]])
    H=np.array(   
[[ 11.20569996e-01 ,-2.21790356e-02 ,-5.38146808e+01],
 [-1.04855958e-01 , 1.01096785e+00,  2.93200271e+01],
 [-2.90820647e-04 , 1.30633807e-04  ,1.00000000e+00]]
 )
    B=np.dot(H,A1)
    cv2.circle(frame0, (int(B[0]), int(B[1])), 1, color[0])

def valor_pixel(id, x, y, frame0,lm, faceLms):
    
    img0_gray= cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)#pasar a escala de grises
    valor_pixel=[]
    ih, iw= img0_gray.shape
    x, y = int(lm.x*iw), int(lm.y*ih)
    if (y <= 160 or x <= 122):
        print ('entra en el else')
        valor_pixel=str(img0_gray[y,x])
    else:
        pass#habra que poner algo aqui
    # print(valor_pixel)

    #Mostrar imagen
    cv2.imshow('imagenGris',img0_gray)
    return valor_pixel
    
    # cv2.putText(frame,'Cristian Garcia',(100,300),2,(255,255,0),2,cv2.LINE_AA)

cap = cv2.VideoCapture(2)#visible
cap0 = cv2.VideoCapture(1)#termica

ancho= cap.get(cv2.CAP_PROP_FRAME_WIDTH)#ancho visible
alto= cap.get(cv2.CAP_PROP_FRAME_HEIGHT)#alto visible
print(ancho)
print(alto)
ancho0= cap0.get(cv2.CAP_PROP_FRAME_WIDTH)#ancho termico
alto0= cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)#alto termico
print(ancho0)
print(alto0)

fps = cap.get(cv2.CAP_PROP_FPS)
time=int(30/fps) #hay que cambiar los ms ya que la camara tiene menos fps que el video anterior
print("Nº de frames por segundo: ",fps)

mpDraw= mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
valor_pixel1=[]
ret, frame = cap.read()
ret0, frame0 = cap0.read()
while(cap.isOpened() and ret): 
    ret, frame = cap.read()
    ret0, frame0 = cap0.read()
    size = (frame.shape[1], frame.shape[0])#redimensión
    frame0= cv2.resize(frame0, size)#redimensión
    imgRGB= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)#cambiar de bgr y rgb
    results=faceMesh.process(imgRGB)
    draw=True
    if results.multi_face_landmarks:
        if draw:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec) #dibujar la cara 
            face=[]    
            for id, lm in enumerate (faceLms.landmark):
                ih, iw, ic= frame.shape
                x, y = int(lm.x*iw), int(lm.y*ih)#Se calcula la posición en píxeles (x,y) de cada landmark en la imagen de entrada (frame).
                # print (id, x,y)
                if (x < 640 or y < 480):#se comprueba
                    face.append([x,y])#ptos de la cara, tienes cad id de punto en una imagen 
                    draw_termica(x,y,frame0)
                    valor_pixel1=valor_pixel(id, x, y, frame0,lm,faceLms.landmark)

    cv2.imshow('Caras ',frame) #crea una ventana de lo que se ve por la cámara junyo a la detetción de rostros
    cv2.imshow("Caras detectadas Termica",frame0)

    if cv2.waitKey(time) & 0xFF == ord('q'):
        cv2.imwrite('visibless__13.jpg', imgRGB)
        cv2.imwrite('termicass__13.jpg', frame0)
        break


cap.release()
cv2.destroyAllWindows()
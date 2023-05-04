# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:20:12 2022

@author: Cristian
"""

## Detección del valor de los puntos de interes de las caras detectdas por el script , 
##se pasa los frames del script final, se detecta la cara en el y se estima el valor medio (color)de los id del rostro térmico imagen térmica

import cv2
import mediapipe as mp
import numpy as np
import time

def fun_valor_pixel0(x,y,img0, valor_pixel, valor_pixel0): #sacar el valor de los pixeles en la iomagen de grises
#sacar valores en forma de X
    for i in range(0,3):
        a=str(img0[(y+i),(x+i)])
        a=int(a)
        if (a >200):
            valor_pixel0.append(a)
            
        a=str(img0[(y+i),(x-i)])
        a=int(a)
        if (a >200):
            valor_pixel0.append(a)
            
        a=str(img0[(y-i),(x-i)])
        a=int(a)
        if (a >200):
            valor_pixel0.append(a)
            
        a=str(img0[(y-i),(x+i)])
        a=int(a)
        if (a >200):
            valor_pixel0.append(a)   
            
    return valor_pixel0

def draw_termica(x,y,img0, valor_pixel, valor_pixel0):

    color=[(0,255,0),(255,0,0),(0,0,255)]
    #color=(255,0,0)
    A1=np.array([[x],[y],[1]])
    H=np.array([[ 11.20569996e-01 ,-2.21790356e-02 ,-5.38146808e+01],
     [-1.04855958e-01 , 1.01096785e+00,  2.93200271e+01],
     [-2.90820647e-04 , 1.30633807e-04  ,1.00000000e+00]])
    B=np.dot(H,A1)
    valor_pixel0=fun_valor_pixel0(int(B[0]), int(B[1]), img0, valor_pixel, valor_pixel0)
    cv2.circle(img0, (int(B[0]), int(B[1])), 1, color[2])

    return valor_pixel0
    
def fun_valor_pixel(x,y,i,img,j,img0, valor_pixel, valor_pixel0):
    
    a=str(img[y,x])
    valor_pixel.append(a)
    j=j+1
    valor_pixel0=draw_termica(x,y,img0, valor_pixel, valor_pixel0)
    return valor_pixel


def estimacion(img, img0):
    # for k in range (36,39):
        valor_pixel=[]
        valor_pixel0=[]
        pixel=[]
        media=None
        # time.sleep(3)
        # k=str(k)
        # visible='visible_estimacion.jpg'
        # termica='termica_estimacion.jpg'
    
        # img = cv2.imread(visible)
        # img0= cv2.imread(termica)
        # size = (img.shape[1], img.shape[0])
        # img0= cv2.resize(img0, size)
        
        j=0

        lower_range = (0, 0, 250)
        upper_range = (255, 255, 255)
        # Convertir el fotograma a HSV
        hsv = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
        
        # Crear la máscara para el rango de temperatura de interés
        mask = cv2.inRange(hsv, lower_range, upper_range)
        
        # Aplicar la máscara al fotograma original para obtener solo los objetos de interés
        img0 = cv2.bitwise_and(img0, img0, mask=mask)
        img0=gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        # Mostrar el fotograma con los objetos de interés
        # cv2.imshow("Máscara sobre rostro", img0)
        
        mpDraw = mp.solutions.drawing_utils
        mpFaceMesh = mp.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
        drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

        
        i=0
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(img)
        
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                      drawSpec,drawSpec)
                landmark=[34,139,71,68,104,69,108,151,337,333,301,368,411,427,187,207]

                for id,lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    # print(id,x,y)
                    
                    if (i==34 or i==139 or i==71 or i==68 or i==104 or i==69 or i==108 or i==151 or i==337 or i==333 or i==301 or i==368 or i==411 or i==427 or i==187 or i==207):
                    # if (i==34 | i==139 | i==71 | i==68 | i==104 | i==69 | i==108 | i==151 | i==337| i==333| i==301| i==368| i==411| i==427| i==187| i==207):
                        # print('El id es', id)
                    
                        pixel.append(fun_valor_pixel(x,y,i,img,j,img0, valor_pixel, valor_pixel0))
                        
                        # print("Color", "fila:", + x, "columna", + y, "=", str(img[x,y]))
                    i=i+1
        # cv2.imshow("Image", img)
        # cv2.imshow("Imagen termica", img0)
        valor_pixel0= list(filter(lambda x: x != "0", valor_pixel0))          
        size=len(valor_pixel0)
        valor_pixel0 = [int(i) for i in valor_pixel0]
        suma = sum(valor_pixel0)
        cantidad = len(valor_pixel0)
        if cantidad !=0:
            media = suma / cantidad
        # print ("la media",k," es ", media)
        # print('estoy') 
            
        # print(valor_pixel0)
        # print(" ")

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if not media:
            return
        return media
        
    # # from keras.models import load_model
    
    # # # Cargar el modelo guardado
    # # model = load_model('mi_modelo1.h5')
    
    # # # Usar el modelo para hacer predicciones
    # ambiente_test=24.1
    # ambiente_test=(ambiente_test- amb_media ) / amb_desvi 
    
    # resultado = model.predict([[(226.31147540983608/255), ambiente_test]])
    # print("El resultado es " + str(resultado) + "ºC")
    # # model.save('mi_modelo1.h5')

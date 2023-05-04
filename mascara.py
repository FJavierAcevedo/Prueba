# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 13:40:56 2023

@author: Cristian
"""

import cv2
#Máscara térmica
# Cargar el video
cap = cv2.VideoCapture(1)
# Definir el rango de temperatura de interés (en este caso, entre 150 y 200 grados)
lower_range = (0, 0, 240)
upper_range = (255, 255, 255)

while True:
    ret, frame = cap.read()
    # Convertir el fotograma a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Crear la máscara para el rango de temperatura de interés
    mask = cv2.inRange(hsv, lower_range, upper_range)
    # Aplicar la máscara al fotograma original para obtener solo los objetos de interés
    result = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Hot Objects", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

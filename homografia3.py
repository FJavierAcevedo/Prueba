# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 13:01:08 2023

@author: Cristian
"""

import cv2

#Script de homografía bueno
# Carga la primera imagen
img1 = cv2.imread("visibless__12.jpg")

size = (img1.shape[1], img1.shape[0])

# Crea una lista para almacenar los puntos de interés seleccionados por el ratón en la primera imagen
points1 = []

# Define la función de devolución de llamada que se ejecutará cada vez que se haga clic con el ratón en la imagen 1
def select_point1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points1.append((x, y))
        # Dibuja un círculo en el punto seleccionado
        cv2.circle(img1, (x, y), 3, (0, 255, 255), -1)
        cv2.imshow("image1", img1)

# Asigna la función de devolución de llamada a la imagen 1
cv2.namedWindow("image1")
cv2.setMouseCallback("image1", select_point1)

# Muestra la imagen 1 y espera a que el usuario seleccione los puntos de interés
while True:
    cv2.imshow("image1", img1)
    key = cv2.waitKey(1)

    # Si el usuario pulsa la tecla 'q', sale del bucle
    if key == ord('q'):
        break

# Carga la segunda imagen
img2 = cv2.imread("termicass__12.jpg")
# img2 = cv2.resize(img2, size)
# Crea otra lista para almacenar los puntos de interés seleccionados por el ratón en la segunda imagen
points2 = []

# Define la función de devolución de llamada que se ejecutará cada vez que se haga clic con el ratón en la imagen 2
def select_point2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points2.append((x, y))
        # Dibuja un círculo en el punto seleccionado
        cv2.circle(img2, (x, y), 3, (0, 255, 255), -1)
        cv2.imshow("image2", img2)

# Asigna la función de devolución de llamada a la imagen 2
cv2.namedWindow("image2")
cv2.setMouseCallback("image2", select_point2)

while True:
    cv2.imshow("image2", img2)
    key = cv2.waitKey(1)

    # Si el usuario pulsa la tecla 'q', sale del bucle
    if key == ord('q'):
        break


import numpy as np

points1 = np.array(points1, dtype=np.float32)
points1 = cv2.UMat(points1)

points2 = np.array(points2, dtype=np.float32)
points2 = cv2.UMat(points2)

# Calcula la matriz de homografía entre los puntos de interés seleccionados
H, _ = cv2.findHomography(points1, points2, method = 0)
print (H)
print('------')
H, _ = cv2.findHomography(points1, points2, method = 4)
print (H)
print('------')
H, _ = cv2.findHomography(points1, points2, method = 8)
print (H)
print('------')
H, _ = cv2.findHomography(points1, points2, method = 16)
print (H)
print('------')
H, _ = cv2.findHomography(points2, points1, method = 0)
print (H)
# Destruye las ventanas abiertas
cv2.destroyAllWindows()

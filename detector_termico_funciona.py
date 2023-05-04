
# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import deteccion_imagen1 as func
import traceback

################# Script final para la estimación de temperatura corporal ######################################

"""
Se obtiene imagen visible e imagen térmica 
Se analiza si en la imagen visible hay un rostro humano con el detector Haar Cascade 
Si se detecta un rostro, se trasladan los pixeles donde está ubicado el rostro en la imagen térmica 
Una vez el rostro es detectado, se aplica una máscara para eliminar los colores fríos que no sean de nuestro interés, esto será muy útil para evitar problemas con posibles gafas o mascarilla que porte el usuario. 
Una vez filtrado el rostro, se pasa al script de promediado de landmark, donde extraerá el valor promedio de los landmark del rostro a partir del modelo mediapipe desarrollado por Google. 
Se consultará el valor obtenido en la red neuronal compilada 
Se mostrará el resultado en el flujo de vídeo 
"""
#Este script realiza un video por la camara del ordenador y cuando se presiona la letra q
# se realiza una captura de lo que se ve en la pantalla
#print ('Nombre vídeo: ', sys.argv[1])

#las variables seguidas de 0, significan que estan orientadas a la camara termica

def draw_rects(img, rects, color,frame0,resultado):
    i=0;
    for x1, y1, x2, y2 in rects:
        color=(255,0,0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color[0], 2) 
        i=i+1
        draw_rects_termica(x1, y1, x2, y2,color,frame0,resultado)
        print("Caras %",rects)#indica la posición de las caras detetctadas
        
def draw_rects_termica(x1, y1, x2, y2,color,frame0,resultado):

    color=(255,0,0)
    A1=np.array([[x1],[y1],[1]])
    H=np.array(   
[[ 11.20569996e-01 ,-2.21790356e-02 ,-5.38146808e+01],
 [-1.04855958e-01 , 1.01096785e+00,  2.93200271e+01],
 [-2.90820647e-04 , 1.30633807e-04  ,1.00000000e+00]])
    B=np.dot(H,A1)
    A2=np.array([[x2],[y2],[1]])
    C=np.dot(H,A2)
    cv2.rectangle(frame0, (int(B[0]), int(B[1])), (int(C[0]), int(C[1])), color[0], 2)
    cv2.putText(frame0, resultado, (int(B[0]),int(B[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) 
    a = int(B[0])
    b = int(B[1])
    c = int(C[0])
    d = int(C[1])
    roi = frame0[b:d, a:c ]
    cv2.imshow('ROI', roi)
    mascara(roi)

    
def mascara (roi):
    
    lower_range = (0, 0, 240)
    upper_range = (255, 255, 255)
    # Convertir el fotograma a HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Crear la máscara para el rango de temperatura de interés
    mask = cv2.inRange(hsv, lower_range, upper_range)
    # Aplicar la máscara al fotograma original para obtener solo los objetos de interés
    result = cv2.bitwise_and(roi, roi, mask=mask)
    # Mostrar el fotograma con los objetos de interés
    cv2.imshow("Máscara sobre rostro", result)
           

def detector(img_color,frame0,acu_media):
    global img_out1 #Se crea como vrble global para el programa
    #Tratamos la imagen visble
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)#convierte de RGB a gris (imagen a cambiar, espacio de color)
    img_gray = cv2.equalizeHist(img_gray)#ecualiza el histograma (en teoría, para que se ve a mejor)
    cv2.imshow("Imagen gris",img_gray)
    
    # tratamos iomagen térmica
    img_gray0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)#convierte de RGB a gris (imagen a cambiar, espacio de color)
    img_gray0= cv2.equalizeHist(img_gray0)#ecualiza el histograma (en teoría, para que se ve a mejor)
    cv2.imshow("Imagen termica gris",img_gray0)
    blur = cv2.blur(img_gray0,(5,5))
    cv2.imshow("Imagen termica suavizada gris",blur)
    
    # Carga del modelo de deteccion (previamente entrenado)
    cascade_fn = 'haarcascades/haarcascade_frontalface_alt.xml'
    cascade = cv2.CascadeClassifier(cascade_fn)
        
    #Llamada al detector
    rects =  cascade.detectMultiScale(img_gray,scaleFactor=1.1,minNeighbors=7,minSize=(30, 30), #estaba minneighbour a 7
                                         flags = cv2.CASCADE_SCALE_IMAGE)
    #minimo nº de pixeles que serán identificados como rostro
    if len(rects) == 0:
        print("Esperando rostro")
        return img_color,frame0
    else:
        #print("Cuadrados",rects)
        rects[:, 2:] += rects[:, :2]   
        color=(0,0,0)
        img_out = img_color.copy()#copia img_color a img_out
        img_out0= frame0.copy()
        resultado=modelo_red_neuronal(frame, frame0,acu_media)
        draw_rects(img_out, rects, color,img_out0,resultado)#llamada a la función para dibujar los rectangulos
        # draw_rects_termica(img_out0, rects, color)
        return img_out,img_out0

def modelo_red_neuronal(frame, frame0, acu_media):
        media=func.estimacion(frame, frame0)
        if media is not None:

            acu_media.append(media)
    
            suma=sum(acu_media)
            length=len(acu_media)
            media2=suma/length
            ambiente_test=21.6
            amb = np.array([20.3,20.3,20.1,20.3,20.1,21.1,24.8,25.7,24.6,24.7,24.7,23.1,23.2,23.3,23.3,23.3,23.3,23.3,23.3,22.1,22.1,22.1], dtype=float)
            amb_media = np.mean(amb)
            amb_desvi = np.std(amb)
            amb_normalizado= (amb - amb_media ) / amb_desvi 
            ambiente_test=(ambiente_test- amb_media ) / amb_desvi 

            resultado = model.predict([[(media2/255), ambiente_test]])
            print("El resultado es " + str(resultado) + "ºC")

            resultado=str(resultado)
            resultado=resultado.replace("[", "").replace("]", "")
            resultado=resultado.replace(",", ".")
            resultado=float(resultado)
            resultado=round(resultado,2)
            resultado=str(resultado)+str(' Celsius +-0.3')

            return resultado
        
        
try:
    from keras.models import load_model
    
    # Cargar el modelo guardado
    model = load_model('mi_modelo2.h5')
    acu_media=[]          
    #Creamos un objeto de la clase video-captura
    cap = cv2.VideoCapture(2)#visible
    cap0 = cv2.VideoCapture(1)#termica
    
    ancho= cap.get(cv2.CAP_PROP_FRAME_WIDTH)#ancho visible
    alto= cap.get(cv2.CAP_PROP_FRAME_HEIGHT)#alto visible
    
    ancho0= cap0.get(cv2.CAP_PROP_FRAME_WIDTH)#ancho termico
    alto0= cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)#alto termico
    
    #Obtenemos la tasa de fps del objeto
    fps = cap.get(cv2.CAP_PROP_FPS)
    #Variable time en ms
    time=int(30/fps) #hay que cambiar los ms ya que la camara tiene menos fps que el video anterior
    print("Nº de frames por segundo: ",fps)
    
    
    #Captura el primer frame
    ret, frame = cap.read() #ret devuelve True si el frame está disponible #frame devuelve la imsgen en vector tu sabe
    ret0, frame0 = cap0.read()
    fourcc=cv2.VideoWriter_fourcc(*'MJPG')
    out=cv2.VideoWriter('videoCristian1.avi',fourcc, 10.0, (640,480))
    
    
    while(cap.isOpened() and ret): 
        size = (frame.shape[1], frame.shape[0])
        frame0= cv2.resize(frame0, size)
        frame00= cv2.GaussianBlur(frame0, (17, 17), 0)
        #cv2.imshow("Video",frame)#crea una ventana de lo que se ve por la cámara
        # img_color=cap.read()
        img_out,img_out0, *c = detector(frame,frame0, acu_media) 
        #Visualizar caras detectadas
        cv2.imshow('Caras detectadas',img_out) #crea una ventana de lo que se ve por la cámara junyo a la detetción de rostros
        cv2.imshow("Caras detectadas Termica",img_out0)#crea una ventana de lo que se ve por la cámara
        cv2.imshow('Caras detectadas suavizadas', frame00)
        
        #cv2.imshow("Caras detectadaaaaas Termica",frame0)
        if cv2.waitKey(time) & 0xFF == ord('q'):
            
            cv2.imwrite('visible_entre_43.jpg', frame)
            cv2.imwrite('termica_entre_43.jpg', frame0)
            cv2.imwrite('visible.jpg', img_out)
            cv2.imwrite('termica.jpg', img_out0)
            print ("Ancho visble: ",ancho)
            print ("Alto visble: ",alto)
            print ("Ancho termica: ",ancho0)
            print ("Alto termica: ",alto0)
            cap.release()
            cap0.release()
            cv2.destroyAllWindows()
            sys.exit()
            break
        if cv2.waitKey(time) & 0xFF == ord('w'):
            modelo_red_neuronal(frame, frame0)
     
        out.write(img_out0)#guarda en el video img_out
        #Captura frame a frame    
        ret, frame = cap.read()
        ret0, frame0 = cap0.read()   
    #out.release()
    cap.release()
    cap0.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
    
finally:
    cap.release()
    cap0.release()
    cv2.destroyAllWindows()
    sys.exit()
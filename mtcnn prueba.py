import cv2
from matplotlib import pyplot
import os
import imutils#cambio de tamaños en imagenes
from mtcnn.mtcnn import MTCNN#descriptor y detector del rostro

# creacion de la carpeta donde almacenara el entrenamieto
direccion = r'C:\Users\josue\Desktop\arquitectura\fotos proyecto\Tapabocas'
nombre = r'Con_tapabocas'
carpeta = direccion + '/' + nombre

# creamos la carpeta 
if not os.path.exists (carpeta):#se crea la carpeta, sin en caso no esta creada
    print("Carpeta creada "+carpeta)
    os.makedirs (carpeta)

#capturamos el video en tiempo real
detector = MTCNN()#el detector es igual a la red neuronal convolucional
cap = cv2.VideoCapture(0) 
count = 0

while True:
    ret, frame = cap.read() 
    gris = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY) 
    copia = frame.copy()

    caras = detector.detect_faces (copia)
    for i in range (len(caras)):#lectura de los rostros detectados
        x1, y1, ancho, alto = caras[i]['box']#coordenadas de esquina superior derecha
        x2, y2 = x1 + ancho, y1 + alto #coordenadas de esquina inferior izquierda
        cara_reg = frame [y1:y2, x1:x2] 
        cara_reg = cv2.resize (cara_reg, (150, 200), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(carpeta +"/rostro_{}.jpg". format (count), cara_reg) 
        count = count + 1
    cv2.imshow("Entrenamiento", frame)
    t = cv2.waitKey(1) 
    if t == 27 or count >= 10:  # salida para esc o llegue al limite de fotos
        break
cap.release() 
cv2.destroyAllWindows()

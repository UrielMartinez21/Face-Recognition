#Importamos las librerias que usaremos
import cv2      #Libreria para la vision artificial
import os       #Realiza operaciones dependientes del Sistema Operativo

#Ruta donde se van a extraer las imagenes
rutaImagenes = "C:/Users/Uriel Martinez/Desktop/Uriel Martinez/Personal/Programacion/Python/Rec_facial/input_faces"

#Si no existe la carpeta entonces se va a crear
if not os.path.exists("faces"):
    os.mkdir("faces")
    print("[+]Nueva carpeta")

#Detector facial
faceClassif=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Contador para guardar los rostros extraidos
count=0

#Leer cada una de las imagenes de la carpeta
for name_imagen in os.listdir(rutaImagenes):
    print(name_imagen)                                      #Imprimira el nombre de cada imagen
    imagen=cv2.imread(rutaImagenes + "/" + name_imagen)     #Ruta donde sacara la imagen
    faces=faceClassif.detectMultiScale(imagen,1.1,5)        #Detecta los rostros
    for(x,y,w,h) in faces:                                  #Obtenemos 4 medidas repecto la imagen
        #cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,255,0),2)  #Visualizamos que ha haya detectado el rostro
        face=imagen[y:y + h, x:x + w]                       #Tomamos el rostro que nos dio en base en ancho y alto
        face=cv2.resize(face,(200,200))                     #Redimencinamos la imagen en 200 px
        cv2.imwrite("faces/" + str(count) + ".jpg", face)   #Guardamos el rostro en la carperta 'faces'
        count+=1

        #cv2.imshow("Faces", imagen)
        #cv2.waitKey(0)

    #1er arg = nombre de la ventana
    #2do arg = ruta de la imagen
    #cv2.imshow("Imagen", imagen)                        #Mostrara la imagen que se guardo
    
    #Con 0 la espera es indefinida hasta el teclado
    #cv2.waitKey(0)                                      #Espera en milisegundo
#cv2.destroyAllWindows()                                 #Destruye las ventanas que se crearon
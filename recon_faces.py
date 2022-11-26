#Importamos las librerias que usaremos
import cv2                  #Libreria para la vision artificial
import os                   #Realiza operaciones dependientes del Sistema Operativo
import face_recognition     #Realiza el reconocimiento de rostros

#Codificacion de rostros extraidos  /   Extraemos vector de 128 elementos
rutaImagenes="C:/Users/Uriel Martinez/Desktop/Uriel Martinez/Personal/Programacion/Python/Rec_facial/faces"

#Arreglos para almacenar informacion
facesEncodig=[]
facesNames=[]

#Leemos cada una de las imagenes dentro de la carpeta
for file_name in os.listdir(rutaImagenes):
    imagen=cv2.imread(rutaImagenes + "/" + file_name)       #Se guarda la ruta de cada imagen
    imagen=cv2.cvtColor(imagen,cv2.COLOR_RGB2BGR)           #Se convierte la imagen a RGB

    #Codificamos cada imagen    /   1erP = nombre de la imagen  /   2doP= Considera toda la imagen
    f_codign= face_recognition.face_encodings(imagen,known_face_locations=[(0,150,150,0)])[0]
    
    #Almacenamos cada imagen codificada
    facesEncodig.append(f_codign)
    #Almacenamos el nombre de cada imagen
    facesNames.append(file_name.split(".")[0])

#Imprimimos la informacion guardada
#print(facesEncodig)
#print(facesNames)


#------------------------------------------------------------------------------

#Variable que abrira la camara de la maquina
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

#Detector facial
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Funcion para abrir la camara
while True:
    ret,frame= cap.read()
    if ret == False:
        break
    
    frame=cv2.flip(frame,1)
    orig= frame.copy()
    faces = faceClassif.detectMultiScale(frame, 1.1, 5) #Detectara los rostros
    
    #Visualizar rostro detectado
    for(x, y, w, h) in faces:
        face=orig[y:y + h,x:x + w]
        face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)   #Conversion a RGB

        #Convertimos el rostro a vector de 128 elementos
        actual_face_encoding=face_recognition.face_encodings(face, known_face_locations=[(0, w, h, 0)])[0]
        
        #Comparamos el rostro actual con el registro
        result=face_recognition.compare_faces(facesEncodig,actual_face_encoding)
        #print(result)                              #Indica en que imagen hay una coincidencia
        if True in result:                          #Traemos el nombre de la coindidencia
            index=result.index(True)
            name=facesNames[index]                  #Se guarda el nombre que coincidio 
            color=(125,220,0)                       #Color verde 
        else:
            name="Desconocido"                      #En caso que no haya coincidencia mostrara el mensaje
            color=(50,50,255)                       #Color rojo

        cv2.rectangle(frame,(x,y + h),(x + w,y + h + 30), color, -1)    #Recuadro verde
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)          #Recuadro ver para el nombre de la persona
        #Mostrar el nombre de la persona sea conocido o no
        cv2.putText(frame, name, (x,y + h + 25), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow("Frame", frame)
    k=cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()               #Libera recurso de software
cv2.destroyAllWindows()     #Se destruyen las ventanas que se crearon 
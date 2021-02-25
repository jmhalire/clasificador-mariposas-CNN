import os #como buscar listar los archivos de un directorio
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models 
import tkinter
from tkinter import *
import tkinter as TK
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

#from sklearn.metrics import confusion_matrix

def Leer_imagenes(carpeta, largo, alto):
    nombres = []
    for r, d, f in os.walk(carpeta):
        for file in f:
            nombres.append(os.path.join(r, file))
    X = np.array([np.array(Image.open(f).resize
                ((largo, alto))) for f in nombres]) / 255.0 #0-255 se vuelve entre 0 y 1
    nombres = np.array([n[len(carpeta)+1:] for n in nombres])
    nombres = np.array([f[:3].title() for f in nombres])
    y,clases = [],[]
    for f in nombres:
        if f not in clases:
            clases.append(f)
        y.append(clases.index(f))
    return X, np.array(y), np.array(clases)

def RNAconvolucional(largo, alto, clases):
    red = models.Sequential()#nombre de la red neuronal
    #primera capa convolucional con activacion RELU y max-pooling 
    #relu funcion relu(x) = max(0, x) 
    red.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(largo, alto, 3)))
    red.add(layers.MaxPooling2D((2, 2)))
    
    #segunda capa convolucional con activacion RELU y max-pooling """
    red.add(layers.Conv2D(64, (3, 3), activation='relu'))
    red.add(layers.MaxPooling2D((2, 2)))
    
    red.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #Aplanar la salida de 4 niveles de la capa convolucional 
    #a 2-rankquen se puede ingresar a una capa totalmente conectada 
    red.add(layers.Flatten())
    #Primera capa completamente conectada con RELU-activacion """
    
    red.add(layers.Dense(256, activation='relu'))
    #Ultima capa totalmente conectada con activacion
    #de softmax para usar en la clasificacion
    #red.add(Dropout(0.5))
    red.add(layers.Dense(clases, activation='softmax'))
    
    #Compilacion del modelo
    red.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return red

def cargar_carpeta():
    carpeta = filedialog.askdirectory()
    try:
        txtDireccion.delete(0, END)
        txtDireccion.insert(0, carpeta)
        global red, clases, largo, alto
        largo, alto = 150, 150 #tamaño de las imagenes
        X, y, clases = Leer_imagenes(carpeta, largo, alto)
        red = RNAconvolucional(largo, alto, len(clases))
        #entrenamiento de la red nueronal
        red.fit(X, y, epochs=50)
        red.save('./modelo/modelo.h5')
        np.savez('./modelo/pesos.npz', clases, (largo, alto))
        messagebox.showinfo("Completado", "Red entrenada y guardada")
    except:
        messagebox.showinfo("Error", "La carpeta no es apropiada")

def CargarModelo():
    try:
    	global red, clases, largo, alto
    	red = models.load_model('./modelo/modelo.h5')
    	parametros = np.load('./modelo/pesos.npz')
    	clases = parametros['arr_0']
    	largo, alto = parametros['arr_1']
    	messagebox.showinfo("Completado", "Modelo Red Neuronal cargada")
    except:
    	messagebox.showinfo("Error", "No se puede recuperar los archivos del modelo")

def Prediccion():
    nombre = filedialog.askopenfilename()
    try:
    	imagen = Image.open(nombre)
    	img = np.array(imagen.resize((largo, alto))) / 255.0
    	imagen = imagen.resize((400, int(imagen.size[1] * 400/imagen.size[0])))
    	imagen = ImageTk.PhotoImage(imagen)
    	contenedor.image = imagen
    	contenedor.create_image(0, 0, anchor=NW, image=imagen)
    	contenedor.place(relx=0.6, rely=0.37)
    except:
    	messagebox.showinfo("Error", "La imagen no puede ser leida")
    try:
    	probabilidades = red.predict_proba(np.array([img]))[0]
    	probabilidades,temp = ["%.2f%%"%(100*f) for f in probabilidades],clases[np.argmax(probabilidades)]
    	#texto = '\n'.join([i+' '*(30-len(i)-len(j))+j for i, j in zip(clases,probabilidades)])
    	texto = '\n'.join(['Esp'+str(i)[2]+' ----> '+j for i, j in zip(clases,probabilidades)])
    	txtPorcentaje['text'] = texto
    	especies=["Danaus plexippus Monarch","Heliconius charitonius Zebra","Heliconius erato Crimson-patched",
                  "Junonia coenia Common","Lycaena phlaeas American","Nymphalis antiopa Mourning","Papilio cresphontes Giant"]
	#txtPredicciones['text'] = texto #+"\n\n\n Es un : "+ str(temp)+especies[int(str(temp[2]))-1]   #str(especies[temp[2]-1])
    	txtEspecie['text'] = "Especie "+str(temp)[2]+" : " +especies[int(str(temp[2]))-1]
    except:
    	messagebox.showinfo("Error", "La red no ha sido creada")
def MatrizDeconfucion():
    textEntrenar= "./data/paraElTest"
    X, y, clases = Leer_imagenes(textEntrenar, 150, 150)
    #red = models.load_model('C:/Users/ASUS/Escritorio/proIA/modelo/modelo.h5')
    #parametros = np.load('C:/Users/ASUS/Escritorio/proIA/modelo/pesos.npz')
    test_eval=red.evaluate(X,y,verbose=1)
    txtExactitud['text'] = "Exactitud: " +str("{0:.2f}".format(float(str(test_eval[1]))*100))+" %"

    #resolviendo matriz de confucion
    mat=[[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
    for r, d, f in os.walk(textEntrenar):
        nombre=[]
        for file in f:
            nombre.append(os.path.join(r, file))
        
    for ele in nombre:
        imagen = Image.open(ele)
        img = np.array(imagen.resize((largo, alto))) / 255.0
        probabilidades = red.predict_proba(np.array([img]))[0]
        temp=clases[np.argmax(probabilidades)]
        mat[int(ele[len(ele)-11:len(ele)-8])-1][int(str(temp))-1] += 1
                
    matriz=""
    for i in range(len(mat)):
        for j in range(len(mat)):
            matriz +=str(mat[i][j])+"\t"
        matriz += "\n"
    txtEspeciesHorizontal['text'] = "Esp1"+"\t"+"Esp2"+"\t"+"Esp3"+"\t"+"Esp4"+"\t"+"Esp5"+"\t"+"Esp6"+"\t"+"Esp7"
    txtEspeciesVertical['text'] = "Esp1"+"\n"+"Esp2"+"\n"+"Esp3"+"\n"+"Esp4"+"\n"+"Esp5"+"\n"+"Esp6"+"\n"+"Esp7"
    txtMatriz['text'] = matriz

cuadro = tkinter.Tk()
cuadro.title("Clasificación de mariposas")
cuadro.geometry('1000x900')
cuadro.configure(background="gray")
txtTitulo = Label(cuadro,text="Clasificación de mariposas",font=("Helvetica", 24),background="silver")
txtTitulo.place(relx=0.35, rely=0.028)

txtBr = Label(cuadro,text='-'*300,font=("Helvetica", 12),background="gray")
txtBr.place(relx=0.0, rely=0.08)

txtBr = Label(cuadro,text='Entrenamiento de red',font=("Helvetica", 14),background="gray")
txtBr.place(relx=0.05, rely=0.12)
#seleccion de la carpeta que contiene la base de datos
btnBuscar = tkinter.Button(cuadro, text = "Seleccionanar DataBase", command=cargar_carpeta,font=("Helvetica", 14),background="black",fg="white")
btnBuscar.place(relx=0.05, rely=0.16)

txtDireccion = Entry(cuadro, width=45, background="thistle2")
txtDireccion.place(relx=0.26,rely=0.165)

#cargarel medelo grabado
txtBr = Label(cuadro,text='Modelo grabado de Red neuronal',font=("Helvetica", 14),background="gray")
txtBr.place(relx=0.7, rely=0.12)

btnRed = tkinter.Button(cuadro, text = "Cargar", command=CargarModelo,font=("Helvetica", 14),background="black",fg="white")
btnRed.place(relx=0.75, rely=0.16)


#matriz de confucion y exactitud
txtBr = Label(cuadro,text='Test de la RNA matriz de confucion y exactittud',font=("Helvetica", 14),background="gray")
txtBr.place(relx=0.12, rely=0.28)

btnpre = tkinter.Button(cuadro, text = "Mostrar", command=MatrizDeconfucion,font=("Helvetica", 14),background="black",fg="white")
btnpre.place(relx=0.12, rely=0.32)

txtEspeciesVertical = Label(cuadro,text='',font=("Helvetica", 14),background="gray")
txtEspeciesVertical.place(relx=0.06, rely=0.45)
txtEspeciesHorizontal = Label(cuadro,text='',font=("Helvetica", 14),background="gray")
txtEspeciesHorizontal.place(relx=0.1, rely=0.41)

txtMatriz = Label(cuadro,text='',font=("Helvetica", 14))
txtMatriz.place(relx=0.1, rely=0.45)

txtExactitud = Label(cuadro,text='',font=("Helvetica", 18),background="green",fg="white")
txtExactitud.place(relx=0.1, rely=0.82)

txtPorcentaje = Label(cuadro,text='',font=("Helvetica", 14),background="gray")
txtPorcentaje.place(relx=0.85, rely=0.40)


#prediccion de una imagen
txtBr = Label(cuadro,text='Prediccion de nueva imagen',font=("Helvetica", 14),background="gray")
txtBr.place(relx=0.6, rely=0.26)

btnImagen = tkinter.Button(cuadro, text = "Seleccionar Imagen", command=Prediccion,font=("Helvetica", 14),background="black",fg="white")
btnImagen.place(relx=0.6, rely=0.30)

contenedor = Canvas(cuadro,width=400,height=400)
contenedor.place(relx=0.6, rely=0.37)

txtEspecie = Label(cuadro,text='',font=("Helvetica", 18),background="green",fg="white")
txtEspecie.place(relx=0.6, rely=0.82)



cuadro.mainloop()

!git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
%pip install -qr requirements.txt  # install

import torch
from yolov5 import utils
display = utils.notebook_init()  # checks

import pandas as pd
from google.colab import drive
from google.colab import files
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os
import glob
from google.colab.patches import cv2_imshow

drive.mount('/content/drive/')
ruta='/content/drive/My Drive/DataSets/Memoria_02/REY_DATASET'

#dtf = pd.read_csv(ruta+"/traza_REY_corregido.csv", sep=";", header=0, encoding='ascii', engine='python')

nom_columnas=["Eliminar1", "Id", "Eliminar2", "Nombre", "Eliminar3", "Origen", "Eliminar4", "Ventana", "Eliminar5", "Rotacion"]



dtf= pd.read_csv(ruta+"/traza_REY_corregido.csv",
                                   sep=";|:",
                                   names=nom_columnas, 
                                   header=None, 
                                   engine="python",
                                   lineterminator='\r',
                                   error_bad_lines=False)

dtf.drop('Eliminar1', axis=1, inplace=True)
dtf.drop('Eliminar2', axis=1, inplace=True)
dtf.drop('Eliminar3', axis=1, inplace=True)
dtf.drop('Eliminar4', axis=1, inplace=True)
dtf.drop('Eliminar5', axis=1, inplace=True)

"""
n_train=int(len(dtf)*0.8)
n_test=int(len(dtf)*0.2)

#print(n_train)
num_aleatorio=random.sample(range(len(dtf)-1), n_test)


dtf_test=[]
dtf_train=[]

for i in range(len(dtf)):
    if i in num_aleatorio:
        dtf_test.append(dtf.iloc[i])
    else:
        dtf_train.append(dtf.iloc[i])
"""

ruta2='/content/drive/My Drive/DataSets/Memoria_02/REY_DATASET/REY_scan_anonim'

#print(ruta2+"/"+dtf["Origen"][0])
#!ls '/content/drive/My Drive/DataSets/Memoria_02/REY_DATASET/REY_scan_anonim'

x=dtf["Ventana"][0].replace('[', '')
x=x.replace(']', '')
x=x.replace(' ', '')
#print(x)
x=x.split(',')
#print(x)

img=cv2.imread(ruta2+"/"+dtf["Origen"][0])

#print(img)
#!ls ruta2
#print(ruta2+"/"+dtf["Origen"][0])
#print(img)
plt.imshow(img)

#print(ruta2+"/"+dtf["Origen"][0])
h, w, c = img.shape

imagenes=[]
ptos=[]
ptos_yolo=[]


num_training=int(0.7*len(dtf))

#print(num_training)

num_aleatorio=random.sample(range(len(dtf)), num_training)

cont_data=0

dtf_training=[]
dtf_resto=[]
for i in range(len(dtf)):
    if i in num_aleatorio:
      dtf_training.append(dtf.iloc[i])
    else:
      dtf_resto.append(dtf.iloc[i])


entreno=ruta2+"/train"
testeo=ruta2+"/test"
validacion=ruta2+"/val"


files = glob.glob(entreno+"/*")
for f in files:
    os.remove(f)

files = glob.glob(testeo+"/*")
for f in files:
    os.remove(f)

files = glob.glob(validacion+"/*")
for f in files:
    os.remove(f)



max=int(len(dtf_resto)//2)
cont=0

imagenes_auxiliares=[]
for i in range(len(dtf)):

  x=dtf["Ventana"][i].replace('[', '')
  x=x.replace(']', '')
  x=x.replace(' ', '')

  x=x.split(',')

  #print(x)
  y=[]



  img = cv2.imread(ruta2+"/"+dtf["Origen"][i])


  #print(img)
  #tratamientos

  trat=2

  if trat==1:
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  
  if trat==2:
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    limite, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)


  imagenes_auxiliares.append(img)

  #print(ruta2+"/"+dtf["Origen"][i])
  #print(img)
  #print(i)
  alto, ancho = img.shape[:2]

  par1=float(x[0])
  par2=float(x[1])
  
  par3=float(x[0])+float(x[3])
  par4=float(x[1])+float(x[2])

  par5=float(x[2])
  par6=float(x[3])

  #print(par1,par2,par3,par4)
  y.append(par1)
  y.append(par2)
  y.append(par3)
  y.append(par4)
  y.append(par5)
  y.append(par6)

  ptos.append(y)

  centroX = round((par1 + par3) / 2)
  CentroY = round((par2 + par4) / 2)


  X1= round(centroX / ancho, 6) 
  X2 = round(CentroY / alto, 6) 
  X3 = round(par6 / ancho, 6) 
  X4 = round(par5 / alto, 6) 


  ptos_yolo_aux=[]
  ptos_yolo_aux.append(X1)
  ptos_yolo_aux.append(X2)
  ptos_yolo_aux.append(X3)
  ptos_yolo_aux.append(X4)

  ptos_yolo.append(ptos_yolo)
  #x="0 "+str(X1)+" "+str(X2)+" "+str(X3)+" "+str(X4)
  #print(x)

  linea_YOLO="0 "+str(X1)+" "+str(X2)+ " " +str(X3)+" "+str(X4)
  
  #print(linea1)
  #print(linea2)

  #print (i, num_aleatorio)
  if i in num_aleatorio:

    pto=ptos_yolo[i]
    

    cv2.imwrite(entreno+"/"+dtf["Origen"][i], img)
    ruta_aux=dtf["Origen"][i].replace('.jpg','.txt')
    ruta_aux=ruta_aux.replace('.png','.txt')
    with open(entreno+"/"+ruta_aux, 'w') as f:
      f.write(linea_YOLO)
      #f.write(linea1)
      #f.write('\n')
      #f.write(linea2)
    f.close()
  else:
    cont=cont+1
    if(cont<=max):
      cv2.imwrite(testeo+"/"+dtf["Origen"][i], img)
      ruta_aux=dtf["Origen"][i].replace('.jpg','.txt')
      ruta_aux=ruta_aux.replace('.png','.txt')
      with open(testeo+"/"+ruta_aux, 'w') as f:
        f.write(linea_YOLO)
        #f.write(linea1)
        #f.write('\n')
        #f.write(linea2)
      f.close()
    else:
      cv2.imwrite(validacion+"/"+dtf["Origen"][i], img)
      ruta_aux=dtf["Origen"][i].replace('.jpg','.txt')
      ruta_aux=ruta_aux.replace('.png','.txt')
      with open(validacion+"/"+ruta_aux, 'w') as f:
        f.write(linea_YOLO)
        #f.write(linea1)
        #f.write('\n')
        #f.write(linea2)   
      f.close()
cont=0
for i in range(len(dtf)):
  #print(dtf["Origen"][i])
  #img=cv2.imread(ruta2+"/"+dtf["Origen"][i])
  #print(imagenes_auxiliares[0])
  pto=ptos[i]
  #print(i, ptos[i])
  img=cv2.rectangle(imagenes_auxiliares[i], (int(pto[0]), int(pto[1]), int(pto[5]), int(pto[4])), (0, 0, 0), 2)

  #print(int(pto[0]), int(pto[1]), int(pto[2]), int(pto[3]))
  #img=cv2.rectangle(img, (5, 5, 180, 180), (0, 0, 0), 2)
  #print(int(x_aa[0]), int(x_aa[1]), int(x_aa[2]), int(x_aa[3]))
  #print(ruta2+"/"+dtf["Origen"][i])
  plt.subplot(1, 4, cont + 1)

  plt.imshow(img, cmap=plt.get_cmap('gray'))
  cont=cont+1
  

  if cont % 4 == 0:
    plt.show()
    plt.clf()
    cont=0
#*******First: crea el archivo data.yaml y ponlo en la carpeta yolov5
#Tiene que tener este formato

%cd /content/yolov5

#Creación del archivo .yaml necesario

ruta_YOLO="/content/yolov5/data"

with open(ruta_YOLO+"/custom.yaml", 'w') as f:
  f.write("train: "+ entreno)
  f.write('\n')
  f.write("val: "+ validacion)
  f.write('\n')
  f.write("test: "+ testeo)
  f.write('\n')
  f.write('\n')
  f.write("# Classes ")
  f.write('\n')
  f.write("nc: 1  # number of classes")
  f.write('\n')
  f.write("names: ['rey']  # class names")
f.close()


%cat data.yaml

"""
train: ../DATASET/images/train/ 
val:  ../DATASET/images/val/
test: ../DATASET/images/test/

nc: 1
names: ['elnombrequequieras']
#####

#Para ver lo que tiene
%cat data.yaml
"""
%cat /content/yolov5/models/yolov5s.yaml
%cat /content/yolov5/models/yolov5s_custom.yaml

import os
os.chdir('/content/yolov5')

#244 es la resolución de la imagen, se toma 244 porque es lo que necesita VGG para funcionar
#Train YOLO
%cd /content/yolov5/
!python train.py --img 244 --batch 80 --epochs 150 --data '/content/yolov5/data/custom.yaml' --cfg '/content/yolov5/models/yolov5s_custom.yaml' --weights ''

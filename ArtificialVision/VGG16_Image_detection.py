import torch
import pandas as pd
from google.colab import drive
from google.colab import files
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os
import glob
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
#from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import load_model
import itertools

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

ruta2='/content/drive/My Drive/DataSets/Memoria_02/REY_DATASET/REY_roi_rot0'

#print(ruta2+"/"+dtf["Origen"][0])
#!ls '/content/drive/My Drive/DataSets/Memoria_02/REY_DATASET/REY_scan_anonim'


img=cv2.imread(ruta2+"/"+dtf["Nombre"][0])


img_claseRot0=[]
img_claseRot90=[]
img_claseRot180=[]
img_claseRotm90=[]


for i in range(len(dtf)):
  if int(dtf["Rotacion"][i])==0:
    img_claseRot0.append(dtf["Nombre"][i])
  elif int(dtf["Rotacion"][i])==90:
    img_claseRot90.append(dtf["Nombre"][i])
  elif int(dtf["Rotacion"][i])==180:
    img_claseRot180.append(dtf["Nombre"][i])
  elif int(dtf["Rotacion"][i])==-90:
    img_claseRotm90.append(dtf["Nombre"][i])

print(len(dtf))
print(len(img_claseRot90))
print(len(img_claseRotm90))
print(len(img_claseRot180))
print(len(img_claseRot0))

entreno_class0=ruta2+"/train/class0"
entreno_class90=ruta2+"/train/class90"
entreno_classm90=ruta2+"/train/classm90"
entreno_class180=ruta2+"/train/class180"

testeo_class0=ruta2+"/test/class0"
testeo_class90=ruta2+"/test/class90"
testeo_classm90=ruta2+"/test/classm90"
testeo_class180=ruta2+"/test/class180"

val_class0=ruta2+"/val/class0"
val_class90=ruta2+"/val/class90"
val_classm90=ruta2+"/val/classm90"
val_class180=ruta2+"/val/class180"


num_t0=int(0.8*(len(img_claseRot0)))
num_tm90=int(0.8*(len(img_claseRotm90)))
num_t90=int(0.8*(len(img_claseRot90)))
num_t180=int(0.8*(len(img_claseRot180)))

num_aleatorio0=random.sample(range(len(img_claseRot0)), num_t0)
num_aleatorio90=random.sample(range(len(img_claseRot90)), num_t90)
num_aleatoriom90=random.sample(range(len(img_claseRotm90)), num_tm90)
num_aleatorio180=random.sample(range(len(img_claseRot180)), num_t180)

img_claseRot0_train=[]
img_claseRot0_resto=[]
img_claseRot90_train=[]
img_claseRot90_resto=[]
img_claseRot180_train=[]
img_claseRot180_resto=[]
img_claseRotm90_train=[]
img_claseRotm90_resto=[]


for i in range(len(img_claseRot0)):
  if i in num_aleatorio0:
    img_claseRot0_train.append(img_claseRot0[i])
  else:
    img_claseRot0_resto.append(img_claseRot0[i])

for i in range(len(img_claseRot90)):
  if i in num_aleatorio90:
    img_claseRot90_train.append(img_claseRot90[i])
  else:
    img_claseRot90_resto.append(img_claseRot90[i])

for i in range(len(img_claseRotm90)):
  if i in num_aleatoriom90:
    img_claseRotm90_train.append(img_claseRotm90[i])
  else:
    img_claseRotm90_resto.append(img_claseRotm90[i])

for i in range(len(img_claseRot180)):
  if i in num_aleatorio180:
    img_claseRot180_train.append(img_claseRot180[i])
  else:
    img_claseRot180_resto.append(img_claseRot180[i])


def cargar_img(ruta, tratamiento):
  img=cv2.imread(ruta)
  #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #limite, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
  return img


print(entreno_class0)


files = glob.glob(entreno_class0+"/*")
for f in files:
    os.remove(f)

files = glob.glob(entreno_class90+"/*")
for f in files:
    os.remove(f)

files = glob.glob(entreno_classm90+"/*")
for f in files:
    os.remove(f)

files = glob.glob(entreno_class180+"/*")
for f in files:
    os.remove(f)


files = glob.glob(testeo_class0+"/*")
for f in files:
    os.remove(f)

files = glob.glob(testeo_class90+"/*")
for f in files:
    os.remove(f)

files = glob.glob(testeo_classm90+"/*")
for f in files:
    os.remove(f)

files = glob.glob(testeo_class180+"/*")
for f in files:
    os.remove(f)


files = glob.glob(val_class0+"/*")
for f in files:
    os.remove(f)

files = glob.glob(val_class90+"/*")
for f in files:
    os.remove(f)

files = glob.glob(val_classm90+"/*")
for f in files:
    os.remove(f)

files = glob.glob(val_class180+"/*")
for f in files:
    os.remove(f)



for i in range(len(img_claseRot0_train)):
  img=cargar_img(ruta2+"/"+img_claseRot0_train[i],1)
  img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  limite, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
  cv2.imwrite(entreno_class0+"/"+img_claseRot0_train[i],img)
              
max=int(len(img_claseRot0_resto)/2)

for i in range(len(img_claseRot0_resto)):
  img=cargar_img(ruta2+"/"+img_claseRot0_resto[i],1)
  if i<=max:
    cv2.imwrite(testeo_class0+"/"+img_claseRot0_resto[i],img)
  else:
    cv2.imwrite(val_class0+"/"+img_claseRot0_resto[i],img)    
              

for i in range(len(img_claseRot90_train)):
  img=cargar_img(ruta2+"/"+img_claseRot90_train[i],1)
  cv2.imwrite(entreno_class90+"/"+img_claseRot90_train[i],img)
              
max=int(len(img_claseRot90_resto)/2)

for i in range(len(img_claseRot90_resto)):
  img=cargar_img(ruta2+"/"+img_claseRot90_resto[i],1)
  if i<=max:
    cv2.imwrite(testeo_class90+"/"+img_claseRot90_resto[i],img)
  else:
    cv2.imwrite(val_class90+"/"+img_claseRot90_resto[i],img)    


for i in range(len(img_claseRotm90_train)):
  img=cargar_img(ruta2+"/"+img_claseRotm90_train[i],1)
  cv2.imwrite(entreno_classm90+"/"+img_claseRotm90_train[i],img)
              
max=int(len(img_claseRotm90_resto)/2)

for i in range(len(img_claseRotm90_resto)):
  img=cargar_img(ruta2+"/"+img_claseRotm90_resto[i],1)
  if i<=max:
    cv2.imwrite(testeo_classm90+"/"+img_claseRotm90_resto[i],img)
  else:
    cv2.imwrite(val_classm90+"/"+img_claseRotm90_resto[i],img)    

for i in range(len(img_claseRot180_train)):
  img=cargar_img(ruta2+"/"+img_claseRot180_train[i],1)
  cv2.imwrite(entreno_class180+"/"+img_claseRot180_train[i],img)
              
max=int(len(img_claseRot180_resto)/2)

for i in range(len(img_claseRot180_resto)):
  img=cargar_img(ruta2+"/"+img_claseRot180_resto[i],1)
  if i<=max:
    cv2.imwrite(testeo_class180+"/"+img_claseRot180_resto[i],img)
  else:
    cv2.imwrite(val_class180+"/"+img_claseRot180_resto[i],img)    

num_classes=4
img_size=224
batch_size_training=100
batch_size_validation=100

data_generator=ImageDataGenerator(preprocessing_function=preprocess_input,)

train_generator=data_generator.flow_from_directory(
	ruta2+"/train",
	target_size=(img_size, img_size),
	batch_size=batch_size_training,
	class_mode='categorical')

validation_generator=data_generator.flow_from_directory(
	ruta2+"/val",
	target_size=(img_size, img_size),
	batch_size=batch_size_validation,
	class_mode='categorical')


model=Sequential()
model.add(VGG16(include_top=False,pooling='avg',weights='imagenet',))

"""
Imagenet es un estándar de facto para la clasificación de imágenes. Se realiza un concurso anual con millones de imágenes de entrenamiento en 1000 categorías. Los modelos utilizados en los concursos de clasificación de 
imagenet se miden entre sí para determinar su rendimiento. Por lo tanto, proporciona una medida "estándar" de qué tan bueno es un modelo para la clasificación de imágenes. Muchos modelos de modelo de aprendizaje de 
transferencia de uso frecuente utilizan los pesos de imagenet. Su modelo, si está utilizando el aprendizaje por transferencia, se puede personalizar para su aplicación agregando capas adicionales al modelo. No tiene 
que usar el peso de imagenet, pero generalmente es beneficioso ya que ayuda a que el modelo converja en menos épocas. Los uso, pero también configuro todas las capas para que se puedan entrenar, lo que ayuda a adaptar 
los pesos del modelo a su aplicación.
"""

model.add(Dense(num_classes, activation='softmax'))

#ver el número de capas de este modelo:
model.layers[0].layers

#este modelo es muy lento, puede tardar horas.

model.layers[0].trainable=False #no queremos reentrenarlos, ya que usamos imagenet

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

steps_per_epoch_train=len(train_generator)
steps_per_epoch_val=len(validation_generator)
number_epoch=50
fit_history=model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch_train, epochs=number_epoch, verbose=1, validation_data=validation_generator, validation_steps=steps_per_epoch_val)

model.save(ruta2+'/modelos_VGG16/VGG16_B&N_Epochs=50.h5')


ruta_pesos=ruta2+'/modelos_VGG16/VGG16_B&N_Epochs=50.h5'
modelo=load_model(ruta_pesos)
testeo=ruta2+"/test"
test_gen=data_generator.flow_from_directory(
	 testeo,
	 target_size=(img_size, img_size),
	 shuffle=False
)

step_per_epoc_test=len(test_gen)

#test_history=modelo.evaluate_generator(test_gen, step_per_epoc_test, verbose=1)
#print("Precisión para el testeo: ", test_history[1])
prediccion=modelo.predict_generator(test_gen,steps=step_per_epoc_test, verbose=1)

from sklearn import metrics

etiquetas=["0 grados", "90 grados", "180 grados", "-90 grados"]

class01=os.listdir(testeo+"/class0")
class02=os.listdir(testeo+"/class90")
class03=os.listdir(testeo+"/class180")
class04=os.listdir(testeo+"/classm90")


valor_verdadero=[]



for i in range(len(class01)):
  valor_verdadero.append(0)
for i in range(len(class02)):
  valor_verdadero.append(1)
for i in range(len(class03)):
  valor_verdadero.append(2)
for i in range(len(class04)):
  valor_verdadero.append(3)


valor_predicho=[]


for i in range(len(prediccion)):
  valor_predicho.append(np.argmax(prediccion[i]))



#print("Recall: " + str(metrics.recall_score(valor_verdadero,valor_predicho,average='weighted')))
cm=metrics.confusion_matrix(valor_verdadero, valor_predicho)


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

plot_confusion_matrix(cm,etiquetas)


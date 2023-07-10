!pip install tensorflow

import torch
import pandas as pd
from google.colab import drive
from google.colab import files
import matplotlib.pyplot as plt
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import random
import os
import glob
from tqdm.notebook  import tqdm
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
#from keras.applications import VGG16
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import VGG19
from sklearn import metrics
import itertools

from keras import layers
from keras.layers import Input, Dense, Activation,BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')
from matplotlib.pyplot import imshow

etiquetas=["Crazing", "Inclusion", "No_defects", "Patches", "Pitted", "Rolled", "Scratches"]
drive.mount('/content/drive/')

ruta="/content/drive/My Drive/DataSets/Material_Defectos"
ruta2="/content/drive/My Drive/DataSets/Material_Defectos/dataset"
print(ruta+"/Etiquetas.csv")

dtf = pd.read_csv(ruta+"/Etiquetas.csv").rename(columns={"Defecto":"Etiqueta"})

#dtf = dtf[dtf["Etiqueta"].isin(etiquetas)].sort_values("id").reset_index(drop=True)

dtf["y"] = dtf["Etiqueta"].factorize(sort=True)[0]

diccionario = dict( dtf[['y','Etiqueta']].drop_duplicates().sort_values('y').values )

print(diccionario)
print(dtf)

redim=(192,192)
#Función para abrir una sola imagen
def load_img(file, ext=['.png','.jpg','.jpeg','.JPG', '.bmp']):
    if file.endswith(tuple(ext)):
        img = cv2.imread(file)
        img=cv2.resize(img, redim, interpolation = cv2.INTER_AREA)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        print("Formato de archivo desconocido.")
        
#Grafica n imagenes en (1 fila) x (n columnas)

def plot_imgs(imgs, titles=[]):
    ## single image
    if (len(imgs) == 1) or (type(imgs) not in [list,pd.core.series.Series]):
        img = imgs if type(imgs) is not list else imgs[0]
        title = None if len(titles) == 0 else (titles[0] if type(titles) is list else titles)
        fig, ax = plt.subplots(figsize=(5,3))
        fig.suptitle(title, fontsize=15)
        if len(img.shape) > 2:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap=plt.cm.binary)
    
    ## multiple imagenes
    else:
        fig, ax = plt.subplots(nrows=1, ncols=len(imgs), sharex=False, sharey=False, figsize=(4*len(imgs),10))
        if len(titles) == 1:
            fig.suptitle(titles[0], fontsize=15)
        for i,img in enumerate(imgs):
            ax[i].imshow(img)
            if len(titles) > 1:
                ax[i].set(title=titles[i])
    plt.show()

ext = ['.png','.jpg','.jpeg','.JPG', '.bmp']

lst_imgs = []
errors = 0

print("*****")
#print(sorted(os.listdir(dirpath)))


lista_aux=os.listdir(ruta+"/dataset/"+"/Crazing")
lista_Crazing=["/Crazing/" + s for s in lista_aux]

lista_aux=os.listdir(ruta+"/dataset/"+"/Inclusion")
lista_Inclusion=["/Inclusion/" + s for s in lista_aux]

lista_aux=os.listdir(ruta+"/dataset/"+"/No_defect")
lista_NoDefect=["/No_defect/" + s for s in lista_aux]

lista_aux=os.listdir(ruta+"/dataset/"+"/Patches")
lista_Patches=["/Patches/" + s for s in lista_aux]

lista_aux=os.listdir(ruta+"/dataset/"+"/Pitted")
lista_Pitted=["/Pitted/" + s for s in lista_aux]

lista_aux=os.listdir(ruta+"/dataset/"+"/Rolled")
lista_Rolled=["/Rolled/" + s for s in lista_aux]

lista_aux=os.listdir(ruta+"/dataset/"+"/Scratches")
lista_Scratches=["/Scratches/" + s for s in lista_aux]

"""
def cargarlista(lista_im):
  lst_imgs=[]
  errors = 0
  for file in tqdm(sorted(lista_im)):
      try:
          if file.endswith(tuple(ext)):
              img = load_img(ruta2+file)
              lst_imgs.append(img)
      except Exception as e:
          print("Fallo del archivo: ", file, "| error:", e)
          errors += 1
          lst_imgs.append(np.nan)
          pass
  return lst_imgs

img_crazing=cargarlista(lista_Crazing)
img_noDefect=cargarlista(lista_NoDefect)
img_inclusion=cargarlista(lista_Inclusion)
img_patches=cargarlista(lista_Patches)
img_pitted=cargarlista(lista_Pitted)
img_rolled=cargarlista(lista_Rolled)
img_scratches=cargarlista(lista_Scratches)
"""                   

entreno=ruta2+"/AlexNet/Normalizado/train"
testeo=ruta2+"/AlexNet/Normalizado/test"
val=ruta2+"/AlexNet/Normalizado/val"

num_crazing=int(0.8*(len(lista_Crazing)))
num_noDefect=int(0.8*(len(lista_NoDefect)))
num_inclusion=int(0.8*(len(lista_Inclusion)))
num_patches=int(0.8*(len(lista_Patches)))
num_pitted=int(0.8*(len(lista_Pitted)))
num_rolled=int(0.8*(len(lista_Rolled)))
num_scratches=int(0.8*(len(lista_Scratches)))


num_aleatorioCrazing=random.sample(range(len(lista_Crazing)), num_crazing)
num_aleatorioNoDefect=random.sample(range(len(lista_NoDefect)), num_noDefect)
num_aleatorioInclusion=random.sample(range(len(lista_Inclusion)), num_inclusion)
num_aleatorioPatches=random.sample(range(len(lista_Patches)), num_patches)
num_aleatorioPitted=random.sample(range(len(lista_Pitted)), num_pitted)
num_aleatorioRolled=random.sample(range(len(lista_Rolled)), num_rolled)
num_aleatorioScratches=random.sample(range(len(lista_Scratches)), num_scratches)

img_crazing_train=[]
img_crazing_resto=[]

img_noDefect_train=[]
img_noDefect_resto=[]

img_inclusion_train=[]
img_inclusion_resto=[]

img_patches_train=[]
img_patches_resto=[]

img_pitted_train=[]
img_pitted_resto=[]

img_rolled_train=[]
img_rolled_resto=[]

img_scratches_train=[]
img_scratches_resto=[]

for i in range(len(lista_Crazing)):
  if i in num_aleatorioCrazing:
    img_crazing_train.append(lista_Crazing[i])
  else:
    img_crazing_resto.append(lista_Crazing[i])

for i in range(len(lista_NoDefect)):
  if i in num_aleatorioNoDefect:
    img_noDefect_train.append(lista_NoDefect[i])
  else:
    img_noDefect_resto.append(lista_NoDefect[i])

for i in range(len(lista_Inclusion)):
  if i in num_aleatorioInclusion:
    img_inclusion_train.append(lista_Inclusion[i])
  else:
    img_inclusion_resto.append(lista_Inclusion[i])

for i in range(len(lista_Patches)):
  if i in num_aleatorioPatches:
    img_patches_train.append(lista_Patches[i])
  else:
    img_patches_resto.append(lista_Patches[i])

for i in range(len(lista_Pitted)):
  if i in num_aleatorioPitted:
    img_pitted_train.append(lista_Pitted[i])
  else:
    img_pitted_resto.append(lista_Pitted[i])

for i in range(len(lista_Rolled)):
  if i in num_aleatorioRolled:
    img_rolled_train.append(lista_Rolled[i])
  else:
    img_rolled_resto.append(lista_Rolled[i])

for i in range(len(lista_Scratches)):
  if i in num_aleatorioScratches:
    img_scratches_train.append(lista_Scratches[i])
  else:
    img_scratches_resto.append(lista_Scratches[i])

def Neighborhood_Operation(img):
  h, w, c = img.shape
  crop_img = img[0:h, 0:w]

def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2

def Contrast_Stretching(img):
  r1 = 66
  s1 = 0
  r2 = 152
  s2 = 255

  pixelVal_vec = np.vectorize(pixelVal)

  contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)

  cv2_imshow(contrast_stretched)


def Grey_level_slicing(img, T1=100, T2=180):

  h,w = img.shape

  img_thresh_back = np.zeros((h,w), dtype = int)

  for i in range(h):
      for j in range(w):  
          if (T1 < img[i,j]).any() and (img[i,j] < T2).any():  
              img_thresh_back[i,j]= 255
          else:
              img_thresh_back[i-1,j-1] = img[i-1,j-1]
  return img_thresh_back


def modificar_gamma(img, tipo, gamma=0.5):

  gamma=gamma
  if tipo==1:
    gamma=0.5
  elif tipo==2:
    gamma=0.7  
  elif tipo==3:
    gamma=0.9 
  elif tipo==4:
    gamma=2.2 
  
  gamma_corrected_img = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
  return(gamma_corrected_img)

def masking(img, tipo):
  if tipo==1:
    kernel = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
          ])
  elif tipo==2:
    kernel = np.array([
          [0, -1, 0],
          [-1, 5, -1],
          [0, -1, 0]
        ])
  elif tipo==3:
    kernel = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
            ]) / 9
  elif tipo==4:
    L = img.max()
    imgNeg = L-img
    cv2_imshow(imgNeg)
    return img
  elif tipo==5:
    kernel = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, 0]]) 
    kernel = 1/3 * kernel
  elif tipo==6:
    kernel = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, 0]]) 
    kernel = 1/3 * kernel
    img=cv2.filter2D(img, -1, kernel)
    img=modificar_gamma(img,2)
    return img
  else:
    return img

  return cv2.filter2D(img, -1, kernel)


def extract_bit_plane(cd):
    #  extracting all bit one by one 
    # from 1st to 8th in variable 
    # from c1 to c8 respectively 
    c1 = np.mod(cd, 2)
    c2 = np.mod(np.floor(cd/2), 2)
    c3 = np.mod(np.floor(cd/4), 2)
    c4 = np.mod(np.floor(cd/8), 2)
    c5 = np.mod(np.floor(cd/16), 2)
    c6 = np.mod(np.floor(cd/32), 2)
    c7 = np.mod(np.floor(cd/64), 2)
    c8 = np.mod(np.floor(cd/128), 2)
    # combining image again to form equivalent to original grayscale image 
    cc = 2 * (2 * (2 * c8 + c7) + c6) # reconstructing image  with 3 most significant bit planes
    img=cc
    to_plot = [cd, c1, c2, c3, c4, c5, c6, c7, c8, cc]
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(10, 8), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for ax, i in zip(axes.flat, to_plot):
        ax.imshow(i, cmap='gray')
    plt.tight_layout()
    plt.show()
    return img
    #enlace: https://www.analyticsvidhya.com/blog/2021/09/a-beginners-guide-to-image-processing-with-opencv-and-python/#:~:text=Installing%20OpenCV%20Package%20for%20Image,of%20programming%20languages%20including%20Python.

def cargar_img(ruta, tratamiento):
  img=cv2.imread(ruta)
  img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  #cv2_imshow(img)

  if tratamiento==0:
    img=img-img.min()
  elif tratamiento==1:
    img=img-img.min()
    img=extract_bit_plane(img)
  elif tratamiento==2:
    img=img-img.min()
    img=masking(img,1)
  elif tratamiento==3:
    img=img-img.min()
    img=masking(img,2)
  elif tratamiento==4:
    img=img-img.min()
    img=masking(img,3)
  elif tratamiento==5:
    img=masking(img,4)
  elif tratamiento==6:
    img=modificar_gamma(img,1)
  elif tratamiento==7:
    img=modificar_gamma(img,2)
  elif tratamiento==8:
    img=modificar_gamma(img,3)
  elif tratamiento==9:
    img=modificar_gamma(img,4)
  elif tratamiento==10:
    img=img-img.min()
    img=Grey_level_slicing(img,10,100)
  elif tratamiento==11:
    img=img-img.min()
    img=255-img
  elif tratamiento==12:
    img=img-img.min()
    img=masking(img,5)
  elif tratamiento==13:
    img=img-img.min()
    img=masking(img,6)

    #img=modificar_gamma(img,5,2)
    #img=Grey_level_slicing(img,10,80)
  #cv2_imshow(img)


  #color_masking(img,1)
  #cv2_imshow(masking(img, 2))

  #waits for user to press any key 
  #(this is necessary to avoid Python kernel form crashing)
  #cv2.waitKey(0) 
    
  #closing all open windows 
  #cv2.destroyAllWindows() 

  #extract_bit_plane(img)

  #limite, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
  return img

n_tratamiento=11
cargar_img(ruta2+img_crazing_train[10],n_tratamiento)
cargar_img(ruta2+img_noDefect_train[10],n_tratamiento)
cargar_img(ruta2+img_inclusion_train[10],n_tratamiento)
cargar_img(ruta2+img_patches_train[10],n_tratamiento)
cargar_img(ruta2+img_pitted_train[10],n_tratamiento)
cargar_img(ruta2+img_rolled_train[10],n_tratamiento)

files = glob.glob(entreno+"/Crazing/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(entreno+"/No_defect/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(entreno+"/Inclusion/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(entreno+"/Patches/"+"/*")
for f in files:
    os.remove(f)


files = glob.glob(entreno+"/Pitted/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(entreno+"/Rolled/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(entreno+"/Scratches/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(testeo+"/Crazing/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(testeo+"/No_defect/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(testeo+"/Inclusion/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(testeo+"/Patches/"+"/*")
for f in files:
    os.remove(f)


files = glob.glob(testeo+"/Pitted/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(testeo+"/Rolled/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(testeo+"/Scratches/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(val+"/Crazing/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(val+"/No_defect/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(val+"/Inclusion/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(val+"/Patches/"+"/*")
for f in files:
    os.remove(f)


files = glob.glob(val+"/Pitted/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(val+"/Rolled/"+"/*")
for f in files:
    os.remove(f)

files = glob.glob(val+"/Scratches/"+"/*")
for f in files:
    os.remove(f)

#mapeo=None
mapeo='gray'

for i in range(len(img_crazing_train)):
  img=cargar_img(ruta2+img_crazing_train[i],n_tratamiento)
  #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #limite, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
  #cv2.imwrite(entreno+img_crazing_train[i],img)
  plt.imsave(entreno+img_crazing_train[i], img, cmap=mapeo)            
max=int(len(img_crazing_resto)/2)

for i in range(len(img_crazing_resto)):
  img=cargar_img(ruta2+img_crazing_train[i],n_tratamiento)
  if i<=max:
    #cv2.imwrite(testeo+img_crazing_train[i],img)
    plt.imsave(testeo+img_crazing_train[i], img, cmap=mapeo) 
  else:
    #cv2.imwrite(val+img_crazing_train[i],img)
    plt.imsave(val+img_crazing_train[i], img, cmap=mapeo)     
  
for i in range(len(img_noDefect_train)):
  img=cargar_img(ruta2+img_noDefect_train[i],n_tratamiento)
  #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #limite, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
  #cv2.imwrite(entreno+img_noDefect_train[i],img)
  plt.imsave(entreno+img_noDefect_train[i], img, cmap=mapeo)     
  #print(ruta2+img_noDefect_train[i])   
  #print(img)
  #print(entreno+img_noDefect_train[i])
max=int(len(img_noDefect_resto)/2)

for i in range(len(img_noDefect_resto)):
  img=cargar_img(ruta2+img_noDefect_train[i],n_tratamiento)
  if i<=max:
    #cv2.imwrite(testeo+img_noDefect_train[i],img)
    plt.imsave(testeo+img_noDefect_train[i], img, cmap=mapeo)
  else:
    #cv2.imwrite(val+img_noDefect_train[i],img)
    plt.imsave(val+img_noDefect_train[i], img, cmap=mapeo)  

for i in range(len(img_inclusion_train)):
  img=cargar_img(ruta2+img_inclusion_train[i],n_tratamiento)
  #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #limite, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
  #cv2.imwrite(entreno+img_inclusion_train[i],img)
  plt.imsave(entreno+img_inclusion_train[i], img, cmap=mapeo)

max=int(len(img_inclusion_resto)/2)

for i in range(len(img_inclusion_resto)):
  img=cargar_img(ruta2+img_inclusion_train[i],n_tratamiento)
  if i<=max:
    #cv2.imwrite(testeo+img_inclusion_train[i],img)
    plt.imsave(testeo+img_inclusion_train[i], img, cmap=mapeo)
  else:
    #cv2.imwrite(val+img_inclusion_train[i],img)  
    plt.imsave(val+img_inclusion_train[i], img, cmap=mapeo)

for i in range(len(img_patches_train)):
  img=cargar_img(ruta2+img_patches_train[i],n_tratamiento)
  #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #limite, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
  #cv2.imwrite(entreno+img_patches_train[i],img)
  plt.imsave(entreno+img_patches_train[i], img, cmap=mapeo)
              
max=int(len(img_patches_resto)/2)

for i in range(len(img_patches_resto)):
  img=cargar_img(ruta2+img_patches_train[i],n_tratamiento)
  if i<=max:
    #cv2.imwrite(testeo+img_patches_train[i],img)
    plt.imsave(testeo+img_patches_train[i], img, cmap=mapeo)
  else:
    #cv2.imwrite(val+img_patches_train[i],img)  
    plt.imsave(val+img_patches_train[i], img, cmap=mapeo)

for i in range(len(img_pitted_train)):
  img=cargar_img(ruta2+img_pitted_train[i],n_tratamiento)
  #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #limite, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
  #cv2.imwrite(entreno+img_pitted_train[i],img)
  plt.imsave(entreno+img_pitted_train[i], img, cmap=mapeo)
              
max=int(len(img_pitted_resto)/2)

for i in range(len(img_pitted_resto)):
  img=cargar_img(ruta2+img_pitted_train[i],n_tratamiento)
  if i<=max:
    #cv2.imwrite(testeo+img_pitted_train[i],img)
    plt.imsave(testeo+img_pitted_train[i], img, cmap=mapeo)
  else:
    #cv2.imwrite(val+img_pitted_train[i],img)  
    plt.imsave(val+img_pitted_train[i], img, cmap=mapeo)

for i in range(len(img_rolled_train)):
  img=cargar_img(ruta2+img_rolled_train[i],n_tratamiento)
  #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #limite, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
  #cv2.imwrite(entreno+img_rolled_train[i],img)
  plt.imsave(entreno+img_rolled_train[i], img, cmap=mapeo)
              
max=int(len(img_rolled_resto)/2)

for i in range(len(img_rolled_resto)):
  img=cargar_img(ruta2+img_rolled_train[i],n_tratamiento)
  if i<=max:
    plt.imsave(testeo+img_rolled_train[i], img, cmap=mapeo)
  else:
    plt.imsave(val+img_rolled_train[i], img, cmap=mapeo)

for i in range(len(img_scratches_train)):
  img=cargar_img(ruta2+img_scratches_train[i],n_tratamiento)
  #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #limite, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
  #cv2.imwrite(entreno+img_scratches_train[i],img)
  plt.imsave(entreno+img_scratches_train[i], img, cmap=mapeo)            
max=int(len(img_scratches_resto)/2)

for i in range(len(img_scratches_resto)):
  img=cargar_img(ruta2+img_scratches_train[i],n_tratamiento)
  if i<=max:
    #cv2.imwrite(testeo+img_scratches_train[i],img)
    plt.imsave(testeo+img_scratches_train[i], img, cmap=mapeo)
  else:
    #cv2.imwrite(val+img_scratches_train[i],img)
    plt.imsave(val+img_scratches_train[i], img, cmap=mapeo)  


#https://dev.to/neeleshrj/image-processing-using-opencv-4kn8


batch_size_training=100
batch_size_validation=50

train__val_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator= train__val_datagen.flow_from_directory(
    entreno, target_size=(200,200), 
    batch_size=batch_size_training,
    class_mode='categorical')


validation_generator=train__val_datagen.flow_from_directory(
	val,
	target_size=(200,200),
	batch_size=batch_size_validation,
	class_mode='categorical')

steps_per_epoch_train=len(train_generator)
steps_per_epoch_val=len(validation_generator)

def AlexNet1(input_shape, num_clases):
    model=keras.models.Sequential([keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=input_shape),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3)),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_clases,activation='softmax')]) 
    return model

def AlexNet2(input_shape, num_clases):
    
    model = keras.Sequential()
    model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), 
                            strides=(4, 4), activation="relu", 
                            input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides= (2, 2)))
    model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), 
                            strides=(1, 1), activation="relu", 
                            padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), 
                            strides=(1, 1), activation="relu", 
                            padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), 
                            strides=(1, 1), activation="relu", 
                            padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), 
                            strides=(1, 1), activation="relu", 
                            padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_clases, activation="softmax"))

    #Añadir capas 

    return model

alex = AlexNet1(train_generator[0][0].shape[1:],7)
alex.summary()

alex.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics=['accuracy'])
alex.fit_generator(train_generator,epochs=50, verbose=1, validation_data=validation_generator, validation_steps=steps_per_epoch_val,steps_per_epoch=steps_per_epoch_train)

ruta_pesos=ruta2+'/AlexNet/modelos/AlexNet_Normalizado_Epochs=50.h5'
alex.save(ruta_pesos)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test = test_datagen.flow_from_directory(val, target_size=(200,200), class_mode='categorical')
preds = alex.evaluate_generator(test)
print ("Loss = " + str(preds[0]))
print ("Val Accuracy = " + str(preds[1]))

predict_datagen = ImageDataGenerator(rescale=1. / 255)
predict = predict_datagen.flow_from_directory(testeo, target_size=(200,200), batch_size = 1,class_mode='categorical')
prediccion = alex.predict_generator(predict)

from sklearn import metrics

class01=os.listdir(testeo+"/Crazing")
class02=os.listdir(testeo+"/Inclusion")
class03=os.listdir(testeo+"/No_defect")
class04=os.listdir(testeo+"/Patches")
class05=os.listdir(testeo+"/Pitted")
class06=os.listdir(testeo+"/Rolled")
class07=os.listdir(testeo+"/Scratches")

valor_verdadero=[]
valor_verdOK_NOK=[]

for i in range(len(class01)):
  valor_verdadero.append(0)
  valor_verdOK_NOK.append(0)
for i in range(len(class02)):
  valor_verdadero.append(1)
  valor_verdOK_NOK.append(0)
for i in range(len(class03)):
  valor_verdadero.append(2)
  valor_verdOK_NOK.append(1)
for i in range(len(class04)):
  valor_verdadero.append(3)
  valor_verdOK_NOK.append(0)
for i in range(len(class05)):
  valor_verdadero.append(4)
  valor_verdOK_NOK.append(0)
for i in range(len(class06)):
  valor_verdadero.append(5)
  valor_verdOK_NOK.append(0)
for i in range(len(class07)):
  valor_verdadero.append(6)
  valor_verdOK_NOK.append(0)




print(len(valor_verdadero))

valor_predicho=[]
valor_predOK_NOK=[]

prediccion=prediccion.tolist()
for i in range(len(prediccion)):

  valor_predicho.append(np.argmax(prediccion[i]))

  if np.argmax(prediccion[i])==2:
    valor_predOK_NOK.append(1)
  else:
    valor_predOK_NOK.append(0)


#print("Recall: " + str(metrics.recall_score(valor_verdadero,valor_predicho,average='weighted')))
cm=metrics.confusion_matrix(valor_verdadero, valor_predicho)
cm_OK_NOK=metrics.confusion_matrix(valor_verdOK_NOK, valor_predOK_NOK)

print(cm_OK_NOK)

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
plot_confusion_matrix(cm_OK_NOK,["KO","OK"])

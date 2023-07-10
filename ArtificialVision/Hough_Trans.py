import sys
import math
import cv2 as cv
import numpy as np
from google.colab import drive
import pandas as pd
import os
from tqdm.notebook  import tqdm
from google.colab.patches import cv2_imshow

etiquetas=["Sano", "DCLev", "DCLest"]
drive.mount('/content/drive/')

ruta="/content/drive/My Drive/DataSets/Memoria_01/Imagenes/Casa"
print(ruta+"/Etiquetas.csv")

dtf = pd.read_csv(ruta+"/Etiquetas.csv").rename(columns={"Defecto":"Etiqueta"})

dtf["y"] = dtf["Etiqueta"].factorize(sort=True)[0]

def load_img(file, ext=['.png','.jpg','.jpeg','.JPG', '.bmp']):
    if file.endswith(tuple(ext)):
        img = cv.imread(file)
        img=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if len(img.shape) > 2:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img
    else:
        print("Formato de archivo desconocido.")

lista_imagenes=os.listdir(ruta+"/sanos")
lista_imagenes=["/sanos/" + s for s in lista_imagenes]

lista_aux=os.listdir(ruta+"/DCL_estables")
lista_aux=["/DCL_estables/" + s for s in lista_aux]

lista_imagenes=lista_imagenes+lista_aux


lista_aux=os.listdir(ruta+"/DCL_evolucion")
lista_aux=["/DCL_evolucion/" + s for s in lista_aux]

lista_imagenes=lista_imagenes+lista_aux

errors = 0
lst_imgs = []
ext = ['.png','.jpg','.jpeg','.JPG', '.bmp']

for file in tqdm(sorted(lista_imagenes)):
    try:
        if file.endswith(tuple(ext)):
            img = load_img(ruta+file)
            lst_imgs.append(img)
    except Exception as e:
        print("Fallo del archivo: ", file, "| error:", e)
        errors += 1
        lst_imgs.append(np.nan)
        pass


dtf["img"] = lst_imgs
dtf = dtf[["Nombre","img","Etiqueta","y"]]

print(dtf.head(300))


dst = cv.Canny(dtf["img"][0], 50, 200, None, 3)
    
# Copy edges to the images that will display the results in BGR
cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)


linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

cv2_imshow(dtf["img"][0])  #cv2_imshow solo se usa en colab para spyder usar cv2.imshow
cv2_imshow(cdst)
cv2_imshow(cdstP)

cv.waitKey()

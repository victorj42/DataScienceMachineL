library(plot3D)
library(readr)
library(wrapr)
library (vtreat)
library(Metrics)
library (dplyr)
library(ggplot2)
library(tidyverse)
library(ggpubr)

ruta <- "D:/Actual/Master IA/2021_2022/Asignaturas/AA/Tarea_01/"

df <- read_delim(paste(ruta,"Data.csv", sep=""), 
                          delim = ";", 
                          col_names = TRUE,
                          escape_double = FALSE, 
                          col_types = cols(Age = col_integer(),
                                           Temp = col_integer(),
                                           Length = col_integer()),  
                          trim_ws = TRUE)

head(df)

#Variables predictoras: Edad (Age) y Temperatura (Temp)
#Estudio de la colinealidad de las variables predictoras
#no se prevee ninguna colinealidad entre la edad y la 
#temperatura, aunque la temperatura pudiera tener una
#influencia en el envejecimiento celular, este no se 
#ha medido, la edad solo indica el paso del tiempo, no el envejecimeinto.
#Por ello no se ve necesario hacer un estudio de colinealidad, aunque
#si se realiza la gráfica de la edad con respecto a la temperatura:

ggplot(df,aes(x=Age,y=Temp))+
  geom_point()

#se puede apreciar, que las mismas edadedes se presentan a las mismas
#temperaturas disponibles, por lo que no hay dependencia.

#Según lo visto anteriormente, parece interesante poner en una gráfica
#la longitud de los peces respecto a la edad coloreandolo por tempertaura.

ggplot(df,aes(x=Age,y=Length, color=Temp))+
  geom_point()

#parece que las gráficas siguen una distribución logarítmica, por lo que
#se visualizará la variable objetivo longitud con respecto al logartimo de 
#la edad

ggplot(df,aes(x=Age,y=Length, color=Temp))+
  geom_point()+
  scale_x_log10()

#hacemos lo propio con respecto a la variable objetivo también

ggplot(df,aes(x=Age,y=Length, color=Temp))+
  geom_point()+
  scale_x_log10()
#Con respecto a la estandarización (normalización de los datos)
#en este caso se usa una regresión lienal por mínimos cuadrados
#que es un método invariante, con lo que la respuesta sustantiva
#no cambia con la estandarización, por lo que no es necesario
#normalizar los datos.

#Dada los resultados gráficos anteriores se decide agrupar los datos por edad obteniendo
#la meida de la temperatura y del tamaño:

df_edad<-df%>%group_by(Age)%>%summarize(mediaT=mean(Temp),mediaLength=mean(Length))
df_edad

#se aprecia que la media de la temperatura es siempre la misma en todos los casos, así que descartaremos
#esta variable como predictor

ggplot(df_edad,aes(x=Age,y=mediaLength))+
  geom_point()+
  scale_x_log10()

modeloLineal <-lm (mediaLength~Age,data=df_edad)
modeloPolGrad2 <- lm(mediaLength~Age+I(Age^2),data=df_edad)
modeloLog <- lm(mediaLength~log(Age),data=df_edad)
r_cuadradas<-c(summary (modeloLineal)$r.squared,summary (modeloPolGrad2)$r.squared,summary (modeloLog)$r.squared)
names(r_cuadradas)<-c("Lineal","Grado2","Logarítmico")




gra_lineal<-ggplot(df_edad,aes(x=Age,y=mediaLength)) +
  geom_point()+
  geom_smooth(method='lm', formula= y~x)+
  annotate("text", x=120, y=5500, label= paste("R^2=",r_cuadradas["Lineal"]),   
             color="blue", size=4,  fontface="bold" )


gra_grad2<-ggplot(df_edad,aes(x=Age,y=mediaLength)) +
  geom_point()+
  geom_smooth(method='lm', formula= y~x+I(x^2))+
  annotate("text", x=120, y=5500, label= paste("R^2=",r_cuadradas["Grado2"]),   
           color="blue", size=4,  fontface="bold" )


gra_log<-ggplot(df_edad,aes(x=Age,y=mediaLength)) +
  geom_point()+
  geom_smooth(method='lm', formula= y~log(x))+
  annotate("text", x=120, y=5500, label= paste("R^2=",r_cuadradas["Logarítmico"]),   
           color="blue", size=4,  fontface="bold" )


ggarrange(gra_lineal, gra_grad2, gra_log + rremove("x.text"), 
          labels = c("     Lineal", "  Pol.Grad.2", " Logarítmica"),
          ncol = 2, nrow = 2)




#Como el modelo mejora algo usando ambas variables, usaramos las dos. Como no 
#tenemos muchos datos para dividir los datos en en conjunto de entrenamiento y
#conjunto de prueba para evaluar el rendimiento del modelo, utilizaremos 
#validación cruzada.

Div_training <- kWayCrossValidation(15,3,NULL,NULL)

k<-3
df$pred.cv <-0
head(df)


for (i in 1:k){
  divide<-Div_training[[i]]
  modelo<-lm(formula = Length~Age, data=df[divide$train,])
  df$pred.cv[divide$app] <- predict(modelo, newdata=df[divide$app, ])
}


modelo=lm(formula = Length_log~Temp_log+Age_log, data=df)

df$pred <- predict(modelo)

rmse(df$pred, df$Length_log)
rmse(df$pred.cv, df$Length_log)


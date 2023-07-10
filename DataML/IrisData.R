library(readr)
library(wrapr)
library (vtreat)
library(Metrics)
library(plot3D)
library(readr)
library(wrapr)
library (vtreat)
library(Metrics)
library (dplyr)
library(ggplot2)
library(tidyverse)
library(ggpubr)
library(purrr)
library(rattle)
library(caret)
library(class)


ruta <- "D:/Actual/Master IA/2021_2022/Asignaturas/AA/Tarea_02/"

df <- read_delim(paste(ruta,"Datos.csv", sep=""), 
                          delim = ";", 
                          col_names = TRUE,
                          escape_double = FALSE, 
                          col_types = cols(Sepal.length = col_double(),
                                           Sepal.width = col_double(),
                                           Petal.length = col_double(),
                                           Petal.width= col_double(),
                                           Species=col_factor()),  
                          trim_ws = TRUE)

df%>% print(n = 100)

plot(df$Petal.length,df$Sepal.length, col=df$Species)
plot(df$Petal.width,df$Sepal.width, col=df$Species)

plot(df$Petal.length,df$Sepal.width, col=df$Species)
plot(df$Petal.width,df$Sepal.length, col=df$Species)



prcomp(df[,c(1:4)], center = TRUE,scale. = FALSE)


distancias<-dist(df[,c(1,2,3,4)], method='euclidean')
modelo_jerarquico<-hclust(distancias, method='ward.D')

asig_clusters<-cutree(modelo_jerarquico, k=3)

plot(modelo_jerarquico,col=df$Species)
plot(asig_clusters, col=df$Species)

predict.hclust(modelo_jerarquico)


modelo_kmean <- kmeans(df[,c(1,2,3,4)], centers=3)

plot(modelo_kmean$cluster, col=df$Species)


N<-nrow(df)
gp<-runif(N)
df<-mutate(df,gp=gp)
head(df)
df_entrenar<-df[df$gp<0.75,]
df_test<-df[df$gp>=0.75,]


resultado_knn<-knn(train=df_entrenar[,c(3,4,2,1)], test=df_test[,c(3,4,2,1)], cl=df_entrenar$Species, k=3)

mean(resultado_knn==df_test$Species)


Val_test<-predict(modelo,new_data=df_test)

formula<-Petal.length~Petal.width
modelo<-lm(formula, data=df_entrenar)


df_entrenar$val_pred<-predict(modelo,new_data=df_entrenar)
df_entrenar$residuales<-resid(modelo)


vreal_test<-df_test$Petal.length   

df_test$pred<-predict(modelo,newdata=df_test)

vpred_test<-df_test$pred  

sqrt(sum((vreal_test-vpred_test)^2)/length(vpred_test))



plot(df_entrenar$Petal.length,df_entrenar$Petal.width)

summary(modelo)

ggplot(df_entrenar,aes(x=Petal.length,y=Petal.width))+ 
  xlab("Longitud")+ 
  ylab("Ancho")+  
  geom_point() +  
  geom_smooth(method=lm, se=FALSE, col='blue', size=2)


ggplot(df_entrenar,aes(x=val_pred,y=residuales))+ 
  xlab("Predichos")+ 
  ylab("Residuales")+  
  geom_point() +  
  geom_hline(yintercept=0)






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
library(tidymodels)
library(ggpubr)
library(purrr)
library(rattle)
library(caret)
library(class)
library(VIM)
library(factoextra)
library(FactoMineR)
library(ggplot2)
library(ggdendro)
library(flashClust)
library(class)
library (naivebayes)
library(tidyverse)
library(rsample)
library(ranger)
library(e1071)
library(tensorflow)
library(keras)
library(psych)
library(dendextend)
library(plotly)
library(ggdendro)
library(reshape)
library(parsnip)
library(dials)
library(tune)
library(yardstick)
library(xgboost)
library(kerasR)


#Cargar los datos del primer conjunto de datos (1992)

ruta <- "D:/Actual/Master IA/2021_2022/Asignaturas/AA/Tarea_03/"
df_1992 <- read_delim(paste(ruta,"breast-cancer.csv", sep=""), 
                  delim = ";", 
                  col_names = TRUE,
                  escape_double = FALSE, 
                  col_types = cols(GrosorG=col_integer(),
                                   Unif_CT = col_integer(),
                                   Unif_CF = col_integer(),
                                   A_marginal = col_integer(),
                                   S_Epi_T = col_integer(),
                                   Nucleos_S = col_integer(),
                                   Crom_B = col_integer(),
                                   Nuecleos_N = col_integer(),
                                   Mitosis = col_integer(),
                                   Tipo = col_integer()),
                  trim_ws = TRUE)

df_1995 <- read.csv(paste(ruta,"wdbc.csv", sep=""))
df_1995$X<-NULL

#visualización de os datos:

df_1992_i<-df_1992


df_1992_i$indice<-as.numeric(row.names(df_1992_i))
names(df_1992_i)[names(df_1992_i) == 'Tipo'] <- 'varO'
df_1992_i$varO<-factor(ifelse(df_1992_i$varO>3,1,0), levels=c(0,1))
df_1992_i$codigo<-NULL
df_1992_i



graficas_dist<-function(datos, cols) {
  
  graf <- data.frame(matrix(ncol = ncol(datos), nrow = 9))
  prednombres<-colnames(datos)
  colnames(graf) <- prednombres
  
  
  for (i in 1:cols){
    graf[[i]]<-local({
      i<-i
      p <- ggplot(data=datos, aes(x=indice, y=datos[,i], color=varO)) + geom_point(size=2, alpha = 0.5)+
        labs(title=prednombres[i])
    })
  }
  return<-graf
}

graficas_caja<-function(datos, cols) {
  
  graf <- data.frame(matrix(ncol = ncol(datos), nrow = 9))
  prednombres<-colnames(datos)
  colnames(graf) <- prednombres
  
  
  for (i in 1:cols){
    graf[[i]]<-local({
      i<-i
      p <- ggplot(data=datos, aes(x=indice, y=datos[,i], color=varO)) + geom_boxplot()+
        labs(title=prednombres[i])
    })
  }
  return<-graf
}

df_1992_i
df_1992_i<-data.frame(df_1992_i)
dibujos<-graficas_dist(df_1992_i, 9)
#dibujos<-graficas_caja(df_1992_i, 9)

graf<-ggarrange(dibujos[[1]],
                    dibujos[[2]],
                    dibujos[[3]],
                    dibujos[[4]],
                    dibujos[[5]],
                    dibujos[[6]],
                    dibujos[[7]],
                    dibujos[[8]],
                    dibujos[[9]],
                    ncol=3, nrow=3)

annotate_figure(graf, top=text_grob("BD breast-cancer-wisconsin 1992", color="blue", size=14, face="bold"))

df_1995_i<-df_1995

#df_1995_i$diagnosis<-factor(ifelse(df_1995_i$diagnosis=="M",1,0), levels=c(0,1))
names(df_1995_i)[names(df_1995_i) == 'diagnosis'] <- 'varO'
ncol(df_1995_i)
df_1995_i_aux=df_1995_i[c(2:31)]
ncol(df_1995_i_aux)
df_1995_i_aux$varO<-df_1995$diagnosis

ncol(df_1995_i_aux)
df_1995_i<-df_1995_i_aux
df_1995_i$varO<-factor(ifelse(df_1995_i$varO=="M",1,0), levels=c(0,1))
ncol(df_1995_i)
df_1995_i$indice<-as.numeric(row.names(df_1995_i))
ncol(df_1995_i)
df_1995_i
ncol(df_1995_i)
dibujos<-graficas(df_1995_i, 30)
#dibujos<-graficas_caja(df_1995_i, 30)
ncol(df_1995_i)

dibujos[[30]]
graf<-ggarrange(dibujos[[1]],
                dibujos[[2]],
                dibujos[[3]],
                dibujos[[4]],
                dibujos[[5]],
                dibujos[[6]],
                dibujos[[7]],
                dibujos[[8]],
                dibujos[[9]],
                dibujos[[10]],
                dibujos[[11]],
                dibujos[[12]],
                dibujos[[13]],
                dibujos[[14]],
                dibujos[[15]],
                dibujos[[16]],
                dibujos[[17]],
                dibujos[[18]],
                dibujos[[19]],
                dibujos[[20]],
                dibujos[[21]],
                dibujos[[22]],
                dibujos[[23]],
                dibujos[[24]],
                dibujos[[25]],
                dibujos[[26]],
                dibujos[[27]],
                dibujos[[28]],
                dibujos[[29]],
                dibujos[[30]],
                ncol=5, nrow=6)

annotate_figure(graf, top=text_grob("Dataset wdbc 1995", color="blue", size=14, face="bold"))

df_1995_i[1:29]

pairs.panels(df_1995_i[1:30], method = "pearson",
             hist.col = "#00AFBB",
             density = TRUE, 
             ellipses = TRUE,
             scale=30
)
df_1992_i
pairs.panels(df_1992_i[1:9], method = "pearson",
             hist.col = "#00AFBB",
             density = TRUE, 
             ellipses = TRUE,
             scale=20
)

#Análisis preliminar de los datos.

#Observamos si hay datos que faltan

sum(is.na(df_1992)) #Faltan 16 datos en la columna Nucleo_s con summary
sum(is.na(df_1995)) #No falta ningún valor

summary(df_1992)
summary(df_1995)
ncol(df_1995)

#vamos a usar la imputación utilizando k vecimos más cercanos, como son 10 posibles valores (enteros [1-10]), 
#vamos a hacer la imputación marcando k=10.

df_1992<-kNN(df_1992, variable="Nucleos_S", k=10)
df_1992$Nucleos_S_imp<-NULL
df_1992$Nucleos_S

#En ambos conjunto de datos la columna de idengificación no nos sirve para nada, así que las eliminamos
df_1992$codigo<-NULL
df_1995$id<-NULL

#Ahora separamos los predictores de la variable objetivo en cada caso:
df_1992
df_1992_p<-as.matrix(df_1992[1:9])
df_1992_o<-df_1992$Tipo
df_1995


df_1995_p<-as.matrix(df_1995[2:31])
df_1995_o<-df_1995$diagnosis


#Según la documentación todos los predictores para df_1992 son valores interos entre 1 a 10, por lo que todos 
#están en la misma escala, sin embargo, no todos los predictores para df_1995 están en la misma escala, por lo 
#que necesitamos escalarlos.

df_1995_pE<-scale(df_1995_p)

#Para que sea más sencilla la categorización de la variable objetivo, factorizamos las variables objetivos en 
#los dos categoría posibles, 0: Negativo (no metástasis), 1: Positivo (sí metástasis).

df_1992_o<-factor(ifelse(df_1992_o>3,1,0), levels=c(0,1))
df_1995_o<-factor(ifelse(df_1995_o=="M",1,0), levels=c(0,1))

#Ya tenemos preparados nuestros dastos para empezar la visulación inicial, sin embargo, considero que ambos
#conjuntos de datos tienen demasiados predictores, es decir, demasiada dimensionalidad, sobretodo a el segundo
#utilizaremos el método PCA para reducir la dimensionalidad de ambos, sin embargo, los estudios en este informe
#se realizarán con 3 conjuntos de datos, el primer conjunto de datos df_1992 sin reducir dimensionalidad, el
#segundo conjunto de datos: df_1992_pca, con los valores de puntajes de los componentes principales obtenidos, y
#df_1995_pca, el segundo conjunto de datos con la reducción de dimensionalidad.
#Empezemos a realizar el análisis PCA para ambos conjuntos de datos, como puede observarse, centramos los datos
#pero no los escalamos, ya que el primer conjunto de datos están a la misma escala, y el segundo ya ha sido escalado
#previamente:

df_1995_pE
df_1992_p
data.frame(df_1995_pE)
ncol(df_1995_pE)
df_1992_pPC<-prcomp(df_1992_p, center=TRUE)
df_1995_pPC<-prcomp(data.frame(df_1995_pE), center=TRUE)
summary(df_1995_pPC)
#obtenemos tanto el gráfico de pantalla de estos dos conjuntos de datos para boservar como podemos reducir la 
#dimensionalidad. 

VARc1992<-df_1992_pPC$sdev^2

pva1992<-data.frame(PC=paste0("PC",1:9), VAR_C=cumsum(VARc1992/sum(VARc1992)))

pva1992%>%ggplot(aes(x=PC, y=VAR_C))+geom_col(fill="blue")+labs(title="Varianza acumulada")+
          geom_hline(yintercept = 0.95, color="red")

#Me quedo con las primeras 7 componentes
df_1992_pca<-df_1992_pPC$x[,1:7]


pva1995<-data.frame(PC=paste0("PC",1:30), VAR_C=cumsum(VARc1995/sum(VARc1995)))

pva1995%>%ggplot(aes(x=fct_reorder(PC,VAR_C), y=VAR_C))+geom_col(fill="blue", width=0.5)+
  labs(title="Varianza acumulada")+
  geom_hline(yintercept = 0.95, color="red")

summary(df_1995_pPC)

df_1995_pca<-df_1995_pPC$x[,1:10]

#Una vez reducido la dimensionalidad del conjunto de datos, tenemos tres conjuntos de datos para 
#poder tratarlos, df_1992_p de 10 predictores, df_1992_pca de 7 predictores, y df_1995_pca de 10 predictores.

#Aprendizaje no supervizado. Una vez tratados los datos, vamos a empezar a comprobar como se agrupan los datos,
#aunque visualmente merece la pena cuando los dos primeros PC acumulan la mayoría de variabiliad. Para estudiar
#el agrupamiento, vamos a usar el clustering gerárquico utilizando el método euclideo para el agrupamiento por
#pares y agrupando según estas distancia siguiendo el método completo.


dist_1992<-dist(df_1992_p, method="euclidean")
dist_1992_pca<-dist(df_1992_pca, method="euclidean")
dist_1995_pca<-dist(df_1995_pca, method="euclidean")

den_1992<-hclust(dist_1992, method="complete")
agr_1992<-cutree(den_1992, k=2)

plot(den_1992, col=df_1992_o)
plot(agr_1992, col=df_1992_o)


den_1992_pca<-hclust(dist_1992_pca, method="complete")
plot(den_1992_pca)

table(agr_1992,df_1992_o)

agr_1992_pca<-cutree(den_1992_pca, k=2)
plot(agr_1992_pca, col=df_1992_o)

table(agr_1992_pca,df_1992_o)


dist_1995_pca

den_1995_pca<-hclust(dist_1995_pca, method="complete")
plot(den_1995_pca)
agr_1995_pca<-cutree(den_1995_pca, k=2)
plot(agr_1995_pca, col=df_1995_o)


dist_1995<-dist(df_1995_p, method="canberra")
den_1995<-hclust((dist_1995)^2, method="ward")
agr_1995<-cutree(den_1995, k=2)
plot(den_1995)
plot(agr_1995, col=df_1995_o)
table(agr_1995, df_1995_o)

#ahora aremos lo propio pero con kmeans.
agr_KM_1992<-kmeans(df_1992_p, centers=2, nstart=30)
plot(agr_KM_1992$cluster, col=df_1992_o)

agr_KM_1992_pca<-kmeans(df_1992_pca, centers=2, nstart=30)
plot(agr_KM_1992_pca$cluster, col=df_1992_o)

agr_KM_1995<-kmeans(df_1995_p, centers=2, nstart=30)
plot(agr_KM_1995$cluster, col=df_1995_o)

agr_KM_1995_pca<-kmeans(df_1995_pca, centers=2, nstart=30)
plot(agr_KM_1995_pca$cluster, col=df_1995_o)

table(agr_KM_1992$cluster, df_1992_o)
table(agr_KM_1992_pca$cluster, df_1992_o)
table(agr_KM_1995$cluster, df_1995_o)
table(agr_KM_1995_pca$cluster, df_1995_o)

agr_KM_1992$cluster
df_1992_o
#aprendizaje supervisado

#Empezamos con KNN

df_1992<-mutate(data.frame(df_1992_p),varO=df_1992_o)
df_1992_PC<-mutate(data.frame(df_1992_pca),varO=df_1992_o)
df_1995_PC<-mutate(data.frame(df_1995_pca),varO=df_1995_o)


planValidacion1<- kWayCrossValidation(nrow(df_1992), 5, NULL, NULL)
planValidacion2<- kWayCrossValidation(nrow(df_1995), 5, NULL, NULL)


k<-5



df_1992_aux<-data.frame(df_1992_p)
df_1992PC_aux<-data.frame(df_1992_pca)
df_1995PC_aux<-data.frame(df_1995_pca)

df_1992_aux$pred.cv<-factor(0, levels=c(0,1))
df_1992PC_aux$pred.cv<-factor(0, levels=c(0,1))
df_1995PC_aux$pred.cv<-factor(0, levels=c(0,1))

df_err_1992<-data.frame(val_k=c(1:7), prec=c(1:7))
df_err_1992PC<-data.frame(val_k=c(1:7), prec=c(1:7))
df_err_1995PC<-data.frame(val_k=c(1:7), prec=c(1:7))


p<-c(1,5,9,13,17,23,27)

for (j in 1:7){
  for(i in 1:k){
    division1<-planValidacion1[[i]]
    division2<-planValidacion2[[i]]
    df_1992_aux[division1$app,]$pred.cv<-knn(train=df_1992[division1$train,], test=df_1992[division1$app,], 
                                             cl=df_1992[division1$train,]$varO, k=p[j])
    df_1992PC_aux[division1$app,]$pred.cv<-knn(train=df_1992_pca[division1$train,], test=df_1992_pca[division1$app,], 
                                               cl=data.frame(df_1992_o)[division1$train,], p[j])
    df_1995PC_aux[division2$app,]$pred.cv<-knn(train=df_1995_pca[division2$train,], test=df_1995_pca[division2$app,], 
                                               cl=data.frame(df_1995_o)[division2$train,], p[j])
  }
  df_err_1992$val_k[j]=p[j]
  df_err_1992$prec[j]=mean(df_1992_o==df_1992_aux$pred.cv)
  
  df_err_1992PC$val_k[j]=p[j]
  df_err_1992PC$prec[j]=mean(df_1992_o==df_1992PC_aux$pred.cv)
  
  df_err_1995PC$val_k[j]=p[j]
  df_err_1995PC$prec[j]=mean(df_1995_o==df_1995PC_aux$pred.cv)
}


ggplot(data=df_err_1992, aes(x=val_k, y=prec))+
  geom_line()+labs(x="Valor de k", y="Precisión") +
  geom_point(size=5, color="red")+
  geom_text(aes(label=paste("k=",as.character(val_k))), hjust=-0.8)


ggplot(data=df_err_1992PC, aes(x=val_k, y=prec))+
  geom_line()+labs(x="Valor de k", y="Precisión") +
  geom_point(size=5, color="red")+
  geom_text(aes(label=paste("k=",as.character(val_k))), hjust=-0.8)

ggplot(data=df_err_1995PC, aes(x=val_k, y=prec))+
  geom_line()+labs(x="Valor de k", y="Precisión") +
  geom_point(size=5, color="red")+
  geom_text(aes(label=paste("k=",as.character(val_k))), hjust=-0.8)



#Ahora vamos a ver que tal se da Bayes:

df_1992c<-df_1992

df_1992c$GrosorG<-factor(df_1992c$GrosorG, levels=c(1,2,3,4,5,6,7,8,9,10))
df_1992c$Unif_CT<-factor(df_1992c$Unif_CT, levels=c(1,2,3,4,5,6,7,8,9,10))
df_1992c$Unif_CF<-factor(df_1992c$Unif_CF, levels=c(1,2,3,4,5,6,7,8,9,10))
df_1992c$A_marginal<-factor(df_1992c$A_marginal, levels=c(1,2,3,4,5,6,7,8,9,10))
df_1992c$S_Epi_T<-factor(df_1992c$S_Epi_T, levels=c(1,2,3,4,5,6,7,8,9,10))
df_1992c$Nucleos_S<-factor(df_1992c$Nucleos_S, levels=c(1,2,3,4,5,6,7,8,9,10))
df_1992c$Crom_B<-factor(df_1992c$Crom_B, levels=c(1,2,3,4,5,6,7,8,9,10))
df_1992c$Nuecleos_N<-factor(df_1992c$Nuecleos_N, levels=c(1,2,3,4,5,6,7,8,9,10))
df_1992c$Mitosis<-factor(df_1992c$Mitosis, levels=c(1,2,3,4,5,6,7,8,9,10))
df_1992c
#nombre_col<-names(df_1992[1:9])
#df_1992c<- lapply(df_1992[,nombre_col] , factor)
#df_1992c$varO<-data.frame(df_1992$varO)


df_1992c_aux<-data.frame(df_1992c)
df_1992PC_aux<-data.frame(df_1992_PC)
df_1992aux<-data.frame(df_1992)



df_1995PC_Taux<-data.frame(df_1995_PC)
df_1995PC_aux<-data.frame(df_1995_PC)



df_1992c_aux$pred.cv<-factor(0, levels=c(0,1))
df_1992aux$pred.cv<-factor(0, levels=c(0,1))
df_1992PC_aux$pred.cv<-factor(0, levels=c(0,1))

df_1995PC_Taux$pred.cv<-factor(0, levels=c(0,1))
df_1995PC_aux$pred.cv<-factor(0, levels=c(0,1))

for(i in 1:k){
  division1<-planValidacion1[[i]]
  division2<-planValidacion2[[i]]
  m_bayes_1992c<-naive_bayes(varO ~ ., data=df_1992c[division1$train,], laplace=1)
  m_bayes_1992<-naive_bayes(varO ~ ., data=df_1992[division1$train,])
  m_bayes_1992PC<-naive_bayes(varO ~ ., data=df_1992_PC[division1$train,])
  m_bayes_1995PC<-naive_bayes(varO ~ ., data=df_1995_PC[division2$train,])
  m_bayes_1995PC_T<-naive_bayes(varO ~ ., data=df_1995_PC[division2$train,], usekernel = T)
  
  df_1992c_aux[division1$app,]$pred.cv<-predict(m_bayes_1992c, df_1992c[division1$app,][1:9])
  df_1992aux[division1$app,]$pred.cv<-predict(m_bayes_1992, df_1992[division1$app,][1:9])
  df_1992PC_aux[division1$app,]$pred.cv<-predict(m_bayes_1992PC, df_1992_PC[division1$app,][1:7])
  df_1995PC_aux[division2$app,]$pred.cv<-predict(m_bayes_1995PC, df_1995_PC[division2$app,][1:10])
  df_1995PC_Taux[division2$app,]$pred.cv<-predict(m_bayes_1995PC_T, df_1995_PC[division2$app,][1:10])
}
df_1992c_aux

mean(df_1992c_aux$pred.cv==df_1992c_aux$varO)
mean(df_1992aux$pred.cv==df_1992aux$varO)
mean(df_1992PC_aux$pred.cv==df_1992PC_aux$varO)
mean(df_1995PC_aux$pred.cv==df_1995PC_aux$varO)
mean( df_1995PC_Taux$pred.cv==df_1995PC_Taux$varO)


#Ahora vamos a probar una regresión logística


umbrales<-seq(0.05,0.85, by=0.1)


u<-length(umbrales)
prec_1992<-data.frame(umb=c(1:u), prec=c(1:u))
prec_1992PC<-data.frame(umb=c(1:u), prec=c(1:u))
prec_1995PC<-data.frame(umb=c(1:u), prec=c(1:u))

for(j in 1:u){
  for(i in 1:k){
    division1<-planValidacion1[[i]]
    division2<-planValidacion2[[i]]
    
    m_RL_1992<-glm(varO ~ ., data=df_1992[division1$train,], family="binomial")
    m_RL_1992PC<-glm(varO ~ ., data=df_1992_PC[division1$train,], family="binomial")
    m_RL_1995PC<-glm(varO ~ ., data=df_1995_PC[division2$train,], family="binomial")
    
   
    df_1992_aux[division1$app,]$pred.cv<-ifelse(predict(m_RL_1992, df_1992[division1$app,][1:9],
                                                        type = "response")>umbrales[j],1,0)
    df_1992PC_aux[division1$app,]$pred.cv<-ifelse(predict(m_RL_1992PC, df_1992_PC[division1$app,][1:7],
                                                        type = "response")>umbrales[j],1,0)
    df_1995PC_aux[division2$app,]$pred.cv<-ifelse(predict(m_RL_1995PC, df_1995_PC[division2$app,][1:10],
                                                        type = "response")>umbrales[j],1,0)
  }
  df_1992_aux$pred.cv
  prec_1992$umb[j]=umbrales[j]
  prec_1992$prec[j]=mean(df_1992_aux$pred.cv==df_1992_o)
  
  prec_1992PC$umb[j]=umbrales[j]
  prec_1992PC$prec[j]=mean(df_1992PC_aux$pred.cv==df_1992PC_aux$varO)
  
  prec_1995PC$umb[j]=umbrales[j]
  prec_1995PC$prec[j]=mean(df_1995PC_aux$pred.cv==df_1995PC_aux$varO)
}
prec_1992
summary(prec_1992)

ggplot(data=prec_1992, aes(x=umb, y=prec))+
  geom_line()+labs(x="umbral de decisión (0-1)", y="Precisión") +
  geom_point(size=5, color="red")+
  geom_text(aes(label=paste("U=",as.character(umb))), hjust=-0.8)

ggplot(data=prec_1992PC, aes(x=umb, y=prec))+
  geom_line()+labs(x="umbral de decisión (0-1)", y="Precisión") +
  geom_point(size=5, color="red")+
  geom_text(aes(label=paste("U=",as.character(umb))), hjust=-0.8)

ggplot(data=prec_1995PC, aes(x=umb, y=prec))+
  geom_line()+labs(x="umbral de decisión (0-1)", y="Precisión") +
  geom_point(size=5, color="red")+
  geom_text(aes(label=paste("U=",as.character(umb))), hjust=-0.8)

prec_1995PC

#ahora vamos a probar un bosque aleatorio

#df_1992A<-df_1992%>%group_by(varO)%>%nest
#df_1992PC_A<-df_1992_PC%>%group_by(varO)%>%nest
#df_1995PC_A<-df_1995_PC%>%group_by(varO)%>%nest

df_1992A_dividir<-initial_split(df_1992, 0.8)
ent_1992A<-training(df_1992A_dividir)
pbr_1992A<-testing(df_1992A_dividir)
ent_1992A

df_1992PC_dividir<-initial_split(df_1992_PC, 0.8)
ent_1992PC<-training(df_1992PC_dividir)
pbr_1992PC<-testing(df_1992PC_dividir)


df_1995PC_dividir<-initial_split(df_1995_PC, 0.8)
ent_1995PC<-training(df_1995PC_dividir)
pbr_1995PC<-testing(df_1995PC_dividir)

v=5
vc_div1992A<-vfold_cv(ent_1992A, v=v)
vc_div1992PC<-vfold_cv(ent_1992PC, v=v)
vc_div1995PC<-vfold_cv(ent_1995PC, v=v)

vc_datos<-vc_div1995PC%>%mutate(entreno=map(splits,~training(.x)), prueba=map(splits,~testing(.x)))


arboles<-c(10,100,500,1000,2000,5000)
nGa<-length(arboles)

bosques<-list()
mtry=c(1,2,3,4,5)
vc_ajuste<-vc_datos%>%crossing(mtry=mtry)

for (a in 1:nGa){

  vc_modelos<-vc_ajuste%>%mutate(modelo=map2(entreno,mtry,~ranger(formula=varO~., 
                                                                data=.x, mtry=.y,num.trees=arboles[a], seed=42)))
  
  
  vc_pred<-vc_modelos%>%mutate(pred=map2(modelo, prueba,~predict(.x,.y)))
  
  
  #mean(vc_pred$pred[[1]]$predictions==vc_pred$prueba[[1]]$varO)
  #vc_pred%>%map2(prueba,pred, ~mean(.x$varO==.y$varO))
  
  
  
  vc_rend<-data.frame(mtry=c(1:length(mtry)), rend=c(1:length(mtry)))
  sub_rend<-0
  
  for (i in 1:length(mtry)){
    df_aux<-vc_pred%>%filter(mtry==mtry[i])
    for(j in 1:v){
      sub_rend<-sub_rend+mean(df_aux$prueba[[j]]$varO==df_aux$pred[[j]]$predictions)
    }
    print(sub_rend)
    vc_rend$mtry[i]=mtry[i]
    vc_rend$rend[i]=sub_rend/v
    sub_rend<-0
  }
  
  bosques<-append(bosques,data.frame(vc_rend))
  
  #bosques$obj[a]<-vc_rend
}


nDatos <- melt(list(A_10 = data.frame(bosques[1],bosques[2]), 
                    A_100 = data.frame(bosques[1],bosques[4]), 
                    A_500 = data.frame(bosques[1],bosques[6]),
                    A_1000 = data.frame(bosques[1],bosques[8]),
                    A_2000 = data.frame(bosques[1],bosques[10]),
                    A_5000 = data.frame(bosques[1],bosques[12])), id.vars = "mtry")
cols <- c("red", "blue", "green", "orange", "gray", "purple")

ggplot(data=nDatos, aes(x=mtry, y=value, colour=L1))+
  geom_line()+labs(x="Valor de mtry", y="Precisión") +
  scale_colour_manual(values=cols)+
  geom_point(size=5, color="black")+
  geom_text(aes(label=paste("M=",as.character(mtry))), hjust=-0.8)



modelo_final<-ranger(formula=varO~.,data=ent_1992A, mtry=2,num.trees=1000, seed=42)

pred<-predict(modelo_final,data=pbr_1992A[1:9])

mean(pred$predictions==pbr_1992A$varO)


modelo_final<-ranger(formula=varO~.,data=ent_1995PC, mtry=2,num.trees=5000, seed=42)

pred<-predict(modelo_final,data=pbr_1995PC[1:10])

mean(pred$predictions==pbr_1995PC$varO)

modelo_final<-ranger(formula=varO~.,data=ent_1992PC, mtry=3,num.trees=2000, seed=42)

pred<-predict(modelo_final,data=pbr_1992PC[1:7])

mean(pred$predictions==pbr_1992PC$varO)

#bosques reforzados
marcador<-boost_tree(trees=2000, 
                     learn_rate=tune(), 
                     tree_depth = tune(), 
                     sample_size = tune(),
                     min_n=tune(), 
                     loss_reduction = tune())%>%
                      set_mode("classification")%>%
                      set_engine("xgboost")

ajusteMar<-marcador%>%extract_parameter_set_dials()%>%grid_random(level=5)


ajuste_resultados<-tune_grid(marcador,
                             varO~.,
                             resamples=vfold_cv(ent_1995PC, v=5),
                             grid=ajusteMar,
                             metrics=metric_set(roc_auc))

autoplot(ajuste_resultados)
mejor_par<-select_best(ajuste_resultados)
spec_final<-finalize_model(marcador,mejor_par)
modelo_final<-spec_final%>%fit(formula=varO~., data=ent_1995PC)
pred<-predict(modelo_final,new_data=pbr_1995PC[1:10])
mean(pred$.pred_class==pbr_1995PC$varO)

errores_1992<-data.frame(arb=c(10,500,1000,1500), prec=c(0.9642857,0.9714286,0.9785714,0.9785714))
errores_1992PC<-data.frame(arb=c(10,500,1000,1500), prec=c(0.9642857,0.9857143,0.9857143,0.9928571))
errores_1995PC<-data.frame(arb=c(10,1000,2000,2500), prec=c(0.9473684,0.9736842,0.9912281,0.9912281))

ggplot(data=errores_1992, aes(x=arb, y=prec, colour="red"))+
  geom_line()+labs(x="Nº de árboles", y="Precisión") +
  geom_point(size=5, color="blue")

ggplot(data=errores_1992PC, aes(x=arb, y=prec, colour="red"))+
  geom_line()+labs(x="Nº de árboles", y="Precisión") +
  geom_point(size=5, color="blue")

ggplot(data=errores_1995PC, aes(x=arb, y=prec, colour="red"))+
  geom_line()+labs(x="Nº de árboles", y="Precisión") +
  geom_point(size=5, color="blue")






#grid_regular(parameters(marcador), levels=3, original = TRUE, filter = NULL)

#ahora vamos a probar svm

sum_err_1992<-0.0
sum_err_1992PC<-0.0
sum_err_1995PC<-0.0

par_1992<-data.frame(gamma,cost)
par_1992PC<-data.frame(gamma,cost)
par_1995PC<-data.frame(gamma,cost)

for(i in 1:k){
    division1<-planValidacion1[[i]]
    division2<-planValidacion2[[i]]

    am_svm_1992<-tune.svm(varO~., data=df_1992[division1$train,], cost=10^(-1:2), 
                     gamma=c(0,1,5,10), kernel="linear")
    
    am_svm_1992PC<-tune.svm(varO~., data=df_1992_PC[division1$train,], cost=10^(-1:2), 
                         gamma=c(0,1,5,10), kernel="linear")
    
    am_svm_1995PC<-tune.svm(varO~., data=df_1995_PC[division2$train,], cost=10^(-1:2), 
                         gamma=c(0,1,5,10), kernel="linear")
    

    mejormodelo_1992<-am_svm_1992$best.model
    pred_1992<-predict(mejormodelo_1992,df_1992[division1$app,][1:9])
    sum_err_1992=sum_err_1992+mean(df_1992[division1$app,]$varO==pred_1992)
    par_1992<-rbind(par_1992,am_svm_1992$best.parameters)
    
    mejormodelo_1992PC<-am_svm_1992PC$best.model
    pred_1992PC<-predict(mejormodelo_1992PC,df_1992_PC[division1$app,][1:7])
    sum_err_1992PC=sum_err_1992PC+mean(df_1992_PC[division1$app,]$varO==pred_1992PC)
    par_1992PC<-rbind(par_1992PC,am_svm_1992PC$best.parameters)
    
    mejormodelo_1995PC<-am_svm_1995PC$best.model
    pred_1995PC<-predict(mejormodelo_1995PC,df_1995_PC[division2$app,][1:10])
    sum_err_1995PC=sum_err_1995PC+mean(df_1995_PC[division2$app,]$varO==pred_1995PC)
    par_1995PC<-rbind(par_1995PC,am_svm_1995PC$best.parameters)
}
sum_err_1992/5
sum_err_1992PC/5
sum_err_1995PC/5

par_1995PC
par_1992PC
par_1992


am_svm_1992<-tune.svm(varO~., data=ent_1992A, cost=c(0.1,10), 
                      gamma=0, kernel="linear")
mejormodelo_1992<-am_svm_1992$best.model
pred_1992<-predict(mejormodelo_1992,pbr_1992A)
mean(pbr_1992A$varO==pred_1992)

am_svm_1992PC<-tune.svm(varO~., data=ent_1992PC, cost=c(0.1,1,10), 
                      gamma=0, kernel="linear")
mejormodelo_1992PC<-am_svm_1992PC$best.model
pred_1992PC<-predict(mejormodelo_1992PC,pbr_1992PC)
mean(pbr_1992PC$varO==pred_1992PC)


am_svm_1995PC<-svm(varO~., data=ent_1995PC, cost=1, 
                   gamma=0, kernel="linear")
pred_1995PC<-predict(am_svm_1995PC,pbr_1995PC)
mean(pbr_1995PC$varO==pred_1995PC)

#Ahora una red profunda

#función de perdida:
#mse: mean_squared_error
#mae: mean_absolute_error
#mape: mean_absolute_percentage_error
#msle: mean_squared_logarithmic_error

############################################################################################################
########################################Base de datos 1992################################################
############################################################################################################
modelo<-keras_model_sequential()
modelo%>%layer_dense(units=9, activation='relu', input_shape = ncol(ent_1992A)-1)%>%
  layer_dense(units=1, activation='sigmoid')


modelo%>%keras::compile(optimizer=optimizer_rmsprop(learning_rate = 0.01), 
                        loss=loss_binary_crossentropy, metrics=c('accuracy','mae'))


x_entreno<-as.matrix(ent_1992A[1:9])
y_entreno<-as.numeric(as.vector(ent_1992A$varO))


modelo%>%keras::fit(x=x_entreno,y=y_entreno, epochs=100,batch_size=30, validation_split=.1, verbose=1)
modelo%>%evaluate(x_entreno, y=y_entreno)
pred<-factor(ifelse(modelo%>%predict(as.matrix(pbr_1992A[1:9]))>0.5,1,0), levels=c(0,1))
mean(pred==pbr_1992A$varO)

############################################################################################################
########################################Base de datos 1992PC################################################
############################################################################################################

modelo<-keras_model_sequential()
modelo%>%layer_dense(units=30, activation='relu', input_shape = ncol(ent_1992PC)-1)%>%
  layer_dense(units=9, activation='relu')%>%
  layer_dense(units=2, activation='relu')%>%
  layer_dense(units=1, activation='sigmoid')

modelo%>%keras::compile(optimizer=optimizer_adamax(learning_rate = 0.01), 
                        loss=loss_binary_crossentropy, metrics=c('accuracy','mae'))


x_entreno<-as.matrix(ent_1992PC[1:7])
y_entreno<-as.numeric(as.vector(ent_1992PC$varO))


modelo%>%keras::fit(x=x_entreno,y=y_entreno, epochs=100,batch_size=30, validation_split=.1, verbose=1)
modelo%>%evaluate(x_entreno, y=y_entreno)
pred<-factor(ifelse(modelo%>%predict(as.matrix(pbr_1992PC[1:7]))>0.5,1,0), levels=c(0,1))
mean(pred==pbr_1992PC$varO)

############################################################################################################
########################################Base de datos 1995PC################################################
############################################################################################################
modelo<-keras_model_sequential()
modelo%>%layer_dense(units=10, activation='relu', input_shape = ncol(ent_1995PC)-1)%>%
            layer_dense(units=1, activation='sigmoid')


modelo%>%keras::compile(optimizer=optimizer_rmsprop(learning_rate = 0.01), 
                        loss=loss_binary_crossentropy, metrics=c('accuracy','mae'))

#modelo%>%compile(optimizer='rmsprop', loss='mse', metrics=c('accuracy', 'mae'))
#modelo%>%compile(optimizer='AdaGrad', loss='mse', metrics=c('accuracy', 'mae'))
nrow(x_entreno)
x_entreno<-as.matrix(ent_1995PC[1:10])
y_entreno<-as.numeric(as.vector(ent_1995PC$varO))

#para claisficación multi clase: y_entreno<-keras::to_categorical(as.matrix(ent_1995PC$varO,num_classes = 2))
y_entreno

modelo%>%keras::fit(x=x_entreno,y=y_entreno, epochs=35,batch_size=30, validation_split=.1, verbose=1)
#modelo%>%keras::fit(x=x_entreno,y=y_entreno, epochs=50, batch_size=30, validation_split=.1, verbose=1)
modelo%>%evaluate(x_entreno, y=y_entreno)
pred<-factor(ifelse(modelo%>%predict(as.matrix(pbr_1995PC[1:10]))>0.5,1,0), levels=c(0,1))
mean(pred==pbr_1995PC$varO)


y_entreno
#binario (0,1): modelo%>%predict(as.matrix(pbr_1995PC[1:10]))%>% `>`(0.5) %>% k_cast("int32")
#multiclase: modelo %>% predict(x) %>% k_argmax()


Sys.setenv(CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin")

tensorflow::tf_config()
callback_tensorboard("logs/run_1")
tensorboard("logs/run_1")


modelo

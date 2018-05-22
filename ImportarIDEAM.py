# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:21:17 2016

@author: GEGEMA 2
"""

'''
Descripción: 
    Este código permite abrir rchivos de Excel y organizar los datos
'''


#------------------
#Paquetes
#------------------

import numpy as np #Manejar matrices
import matplotlib.pyplot as plt# Graficar
import netCDF4 as nc #abrir archivos netCDF
import scipy.io as sio #Para guaradr archivo en Matlab
from datetime import date, timedelta
import xlrd #para arir archvos de excel
from datetime import datetime
import pandas as pd


#--------------------
#Se extraen los datos
#-----------------------

Name= 'C:/Users/Unalmed/Documents/Codigos/IDEAM/DIEGOLEONLH16-VELO.xlsx'


#------------------------
#se nombran as estaciones
#------------------------

#Se abre el archivo

MisDatos=xlrd.open_workbook(Name)
#for hoja in range(1,4):
    
HojaActual= MisDatos.sheet_by_index(0)
    
    
    #Se extraelacolumna 2  para encontrar indicadores de Enero
Columna0= []
Columna1= []
Columna2= []
Columna3= []
Columna10= []
Mes=[]
Year=[]


   
i=0
while True:
    try:
        Columna0.append(HojaActual.cell(i,0).value)
        Columna1.append(HojaActual.cell(i,1).value)
        Columna2.append(HojaActual.cell(i,2).value)
        Columna3.append(HojaActual.cell(i,3).value)
        Columna10.append(HojaActual.cell(i,10).value)
        Mes.append(HojaActual.cell(i,7).value)
        Year.append(HojaActual.cell(i,8).value)
        i+=1
    except IndexError:
        break
        
Columna0=np.array(Columna0)  
Columna1=np.array(Columna1)    
Columna2=np.array(Columna2)
Columna3=np.array(Columna3)
Columna10=np.array(Columna10)
Mes=np.array(Mes)    
Year=np.array(Year)   

    

n = 30     #28 dias como minimo, mas dos espacios en blanco
y = [10,21]  #espacios en blanco
x = set(range(n)) - set(y)
x=np.array(list(x))


rdias=[]
for i in np.arange(1.0,32.0):
    rdias.append(str(i))

velcalm=[]
for i in range (len(Columna1)):
    if Columna1[i]=='VELOCIDAD':
        for j in range(10):
            velcalm.append(i+j)
        
    
dias=[]
posiciones=[]
for i in range (len(Columna1)):
    if (i not in velcalm):
        if (str(Columna0[i]) or str(Columna1[i]) in rdias):
            if Columna0[i] in rdias:
                dias.append(int(float(str(Columna0[i]))))
                posiciones.append(i)
            elif Columna1[i] in rdias:
                dias.append(int(float(str(Columna1[i]))))
                posiciones.append(i)
        
posiciones=np.array(posiciones)        
        
parte=[]
for i in range (len(Columna10)):
    if Columna10[i]=='PARTE)':
        parte.append(i)
        

parte1=[]
parte2=[]

for j in (range(0,198,2)):
    for i in range (len(Columna10)):
        cont=j
        if (i > parte[cont]) & (i<parte[cont+1]):
            parte1.append(i)
        cont+=1
        if (i > parte[cont]) & (i<parte[cont+1]):
            parte2.append(i)
    
 #ciclo para recuperar los ultimos datos u completar la lista anterior           
for i in range (len(Columna10)):          
    if (i >parte[-2]) &  (i <parte[-1]):
           parte1.append(i)
    elif (i >parte[-1]):
           parte2.append(i)



DATOSVEL=[]
posi=[]

for i in posiciones:
    if i in parte1:
        for j in range (3,31,2):
            DATOSVEL.append(str(HojaActual.cell(i,j).value))
            posi.append(i)
    if i in parte2:
        for k in range (3,23,2):
            DATOSVEL.append(str(HojaActual.cell(i,k).value))
            posi.append(i)
        
DATOSVEL=np.array(DATOSVEL)    
posi=np.array(posi)

direcciones=['N','NE','NW','S','SE','SW','W','E', 'C']

XX=[]###VALORES ERRADOS
for i in range(len(DATOSVEL)):
    if DATOSVEL[i] in direcciones:
        XX.append(i)

          
posi[XX]  #####PARA VER EN EL EXCEL DONDE ESTAN

DATOSDIR=[]
for i in posiciones:
    if i in parte1:
        for j in range (2,30,2):
            DATOSDIR.append(str(HojaActual.cell(i,j).value))
    if i in parte2:
        for k in range (2,22,2):
            DATOSDIR.append(str(HojaActual.cell(i,k).value))
            
        
DATOSDIR=np.array(DATOSDIR)    

YY=np.where(DATOSDIR=='93.0')

VALORESNULOSDIR=DATOSDIR[XX]


DATOSVEL[XX]=np.NAN 
DATOSDIR[XX]=np.NAN
DATOSVEL[DATOSVEL=='']=np.NAN 
DATOSDIR[DATOSDIR=='']=np.NAN


VEL=[]
for i in range (len(DATOSVEL)):
    VEL.append(float(DATOSVEL[i]))
    
VEL=np.array(VEL)

DIR=np.array(DATOSDIR)


####LEER FECHAS
VIENTOTIME=[]
for i in range (len(Columna2)):
    if Columna2[i]=='VIENTO':
        VIENTOTIME.append(i)


TIME=[]
for i in  VIENTOTIME:
    TIME.append((HojaActual.cell(i,7).value))
TIME=np.array(TIME)    
    


Meses=['ENE', 'FEB','MAR','ABR','MAY','JUN','JUL','AGO','SEP','OCT','NOV','DIC']
Meses2=['01', '02','03','04','05','06','07','08','09','10','11','12']

MES=[]
for i in TIME:
    for j in range(len(Meses)):
        if str(i) == Meses[j]:
            MES.append(Meses2[j])
            
       

YEAR=[]
for i in  VIENTOTIME:
    YEAR.append(int((HojaActual.cell(i,8).value)))
    
YEAR=np.array(YEAR)    
   
year=YEAR[np.arange(0,200,2)]    
  
f=0
anos=[]
for i in range(len(dias)-1):
    if dias[i+1]>dias[i]:
        anos.append(YEAR[f])
    elif dias[i+1]<dias[i]:
        anos.append(YEAR[f])
        f+=1
        

anos.append(YEAR[-1])
Anos=np.array(anos)


f=0
meses=[]
for i in range(len(dias)-1):
    if dias[i+1]>dias[i]:
        meses.append(MES[f])
    elif dias[i+1]<dias[i]:
        meses.append(MES[f])
        f+=1
        
        
meses.append(MES[-1])
MESES=np.array(meses)

FECHAS=[]


for t,i in enumerate(posiciones):
    
    
    if i in parte1:
        for j in range (0,14):
                FECHAS.append(str(Anos[t])+'/'+str(MESES[t])+'/'+str(dias[t])+'/'+str(j))
    if i in parte2:
        for m in range (14,24):
                FECHAS.append(str(Anos[t])+'/'+str(MESES[t])+'/'+str(dias[t])+'/'+str(m))
   
FECHAS=np.array(FECHAS)

  
Misfechas=[]
for i in FECHAS:
    Misfechas.append(datetime.strptime(i, '%Y/%m/%d/%H'))

WIND_VEL=pd.DataFrame(VEL, index=Misfechas)
WIND_DIR=pd.DataFrame(DIR, index=Misfechas)   
WIND_VEL=WIND_VEL.sort_index()    
WIND_DIR=WIND_DIR.sort_index()  

  
Dates = pd.date_range('2001-01-01-00', '2012-01-31-23', freq='H')
Dates2=np.array(Dates)

Ideam_vel=[]
for i in range(len(Dates)):
    if Dates[i] in WIND_VEL.index:
        j=(np.where(WIND_VEL.index==Dates[i])[0][0])
        Ideam_vel.append(WIND_VEL[0][j])
    else:
        Ideam_vel.append(np.nan)
       
        
Ideam_dir=[]
for i in range(len(Dates)):
    if Dates[i] in WIND_DIR.index:
        j=(np.where(WIND_DIR.index==Dates[i])[0][0])
        Ideam_dir.append(WIND_DIR[0][j])
    else:
        Ideam_dir.append(np.nan)
Ideam_vel=np.array(Ideam_vel)  
Ideam_vel[Ideam_vel>20]=np.NAN    
VEL_IDEAM=pd.DataFrame(Ideam_vel,index=Dates)
DIR_IDEAM=pd.DataFrame(Ideam_dir,index=Dates)

###se remueven valores 




test=VEL_IDEAM[VEL_IDEAM.index.year==2008]



    
    
   
    
    #Se guarda la info
  #  Nameout=Pathout+ Names[hoja-1]+'.mat'
sio.savemat('DatosIdeamVEL.mat',{'FECHA':Dates2, 'VELOCIDAD':Ideam_vel})
    
import scipy.io as sio # Para guardar archivos de Matlab     
mat = sio.loadmat('DatosIdeamVEL.mat')
WindV = mat['VELOCIDAD']
Time = mat['FECHA']            
                     
        
        
        
        


















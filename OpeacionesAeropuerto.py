#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 08:27:56 2016

@author: DGD042
"""


import numpy as np # Manejar matrices
import matplotlib.pyplot as plt # Graficar
import netCDF4 as nc # Abrir archivos NetCDF
import scipy.io as sio # Para guardar archivos de Matlab
from datetime import date, timedelta
from scipy import stats as st
import xlsxwriter as xlsxwl # Crear archivos de Excel

Path = 'C:/Users/UNALMED/Documents/ULTIMO/era/' 
Pathout = 'C:/Users/UNALMED/Documents/ULTIMO/'

mat = sio.loadmat(Path+'VientoZonal.mat')
mat2 = sio.loadmat(Path+'Humedad_Especifica.mat')
mat3 = sio.loadmat(Path+'Apto_Sesquicent.mat')


# Se crea el vector de fechas del Reanalisis ERA-Interim
FechaRean = []
for i in range(1979,2008+1):
    for j in range(1,13):
        FechaRean.append(date(i,j,1))

FechaReanstr = [i.strftime('%Y'+'/'+'%m'+'/'+'%d') for i in FechaRean]





U = mat['u_zonal']
q = mat2['q']
Uxq = U*q





# Se llama el vector de preciptiación
P = mat3['Prec'][0]
FechaP = mat3['FechaC']
# Se recotra el vector de fechas
Fechai = np.where(FechaP == '1979/01/01')[0]
Fechaf = np.where(FechaP == '2008/12/31')[0]

FechaPR = FechaP[3652:14610]
PR=P[Fechai:]


FechaM = np.array([i[:7] for i in FechaPR])

Ai = int(FechaPR[0][:4])
Af = int(FechaPR[-1][:4])







# Inicializar variables
PM = []
FechaMen = []



# Se sacan los promedios mensuales
for i in range(Ai,Af+1): # Ciclo para los años
    for j in range(1,13):
        if j < 10:
            AM = str(i)+'/'+'0'+str(j)
        else:
            AM = str(i)+'/'+str(j)
     
            
         
        x = np.where(FechaM == AM)[0]
        
        # Promedios mensuales
        # Se eliminan meses incompletos
        
        nan = np.isnan(PR[x])
        nanC = np.sum(nan)
        if nanC >= 0.3*len(PR[x]):
            PM.append(np.nan)
        else:
            PM.append(np.nansum(PR[x]))
        
        FechaMen.append(date(i,j,1))
        
        

      
# Se encuentra la calidad de la información
PM = np.array(PM)
nan = np.isnan(PM)
Tnan = np.sum(nan)/len(PM)
TReal = 1-Tnan


U=np.array(U)




#Se obtiene el promedio regional
U_prom=np.mean(U,axis=(1,2))
q_prom=np.mean(q,axis=(1,2))
Uxq_prom=np.mean(Uxq,axis=(1,2))



#Se arma la matriz de datos mensueales

PMM=np.reshape(PM,(-1,12))
UMM=np.reshape(U_prom,(-1,12))
qMM=np.reshape(q_prom,(-1,12))
UqMM=np.reshape(Uxq_prom,(-1,12))

# Se calcula el promedio mensual multianual Precip
PMMan = np.nanmean(PMM,axis=0)
PMDes = np.nanstd(PMM,axis=0)

PEst = []
# Ciclo para calcular los datos estandarizados P
for j in range(PMM.shape[0]):
    for i in range(12): # ciclo para los meses
        PEst.append((PMM[j,i]-PMMan[i])/PMDes[i])
        
        
# Se calcula el promedio mensual multianual Uwind
UMMan = np.nanmean(UMM,axis=0)
UMDes = np.nanstd(UMM,axis=0)

UEst = []
# Ciclo para calcular los datos estandarizados
for j in range(UMM.shape[0]):
    for i in range(12): # ciclo para los meses
        UEst.append((UMM[j,i]-UMMan[i])/UMDes[i])       
        
# Se calcula el promedio mensual multianual humedad especifica
qMMan = np.nanmean(qMM,axis=0)
qMDes = np.nanstd(qMM,axis=0)

qEst = []
# Ciclo para calcular los datos estandarizados
for j in range(qMM.shape[0]):
    for i in range(12): # ciclo para los meses
        qEst.append((qMM[j,i]-qMMan[i])/qMDes[i])       
 
        
# Se calcula el promedio mensual multianual humedad especifica x u
UqMMan = np.nanmean(UqMM,axis=0)
UqMDes = np.nanstd(UqMM,axis=0)

UqEst = []
# Ciclo para calcular los datos estandarizados
for j in range(UqMM.shape[0]):
    for i in range(12): # ciclo para los meses
        UqEst.append((UqMM[j,i]-UqMMan[i])/UqMDes[i])    
        
        
        





             # Se obtienen los promedios trimestrales PM datos estand
# Se obtienen los promedios trimestrales
EPTri = [np.nanmean(PEst[:2])]
EUTri = [np.nanmean(UEst[:2])]
EqTri = [np.nanmean(qEst[:2])]
EUqTri = [np.nanmean(UqEst[:2])]
        
PEst=np.array(PEst)



for i in range(2,PEst.shape[0]-1,3):
    print(i+1)
    EPTri.append(np.nanmean(PEst[i:i+3]))
    EUTri.append(np.nanmean(UEst[i:i+3]))
    EqTri.append(np.nanmean(qEst[i:i+3]))
    EUqTri.append(np.nanmean(UqEst[i:i+3]))
    
EPTriM = np.reshape(EPTri,(-1,4))
EUTriM = np.reshape(EUTri,(-1,4))
EqTriM = np.reshape(EqTri,(-1,4))
EUqTriM = np.reshape(EUqTri,(-1,4))


# Se arma la matriz de correlación trimestrales de P con U EST
ECorrPvU = np.zeros((4,4))
ESigPvU = np.zeros((4,4))

for i in range(4):
    # Se obtienen las correlaciones simultáneas
    # Se eliminan los nan
    qq = ~(np.isnan(EPTriM[:,i])|np.isnan(EUTriM[:,i]))
    ECorrPvU[i,i] = st.pearsonr(EPTriM[qq,i],EUTriM[qq,i])[0]
    ESigPvU[i,i] = st.pearsonr(EPTriM[qq,i],EUTriM[qq,i])[1] 
    for j in range(4):
        if j > i:
            qq = ~(np.isnan(EPTriM[:,j])|np.isnan(EUTriM[:,i]))
            ECorrPvU[j,i] = st.pearsonr(EPTriM[qq,j],EUTriM[qq,i])[0]
            ESigPvU[j,i] = st.pearsonr(EPTriM[qq,j],EUTriM[qq,i])[1]
        if j < i:
            qq = ~(np.isnan(EPTriM[1:,j])|np.isnan(EUTriM[:-1,i]))
            ECorrPvU[j,i] = st.pearsonr(EPTriM[1:,j][qq],EUTriM[:-1,i][qq])[0]
            ESigPvU[j,i] = st.pearsonr(EPTriM[1:,j][qq],EUTriM[:-1,i][qq])[1]


# Se arma la matriz de correlación trimestrales de P con q EST
ECorrPvq = np.zeros((4,4))
ESigPvq = np.zeros((4,4))

for i in range(4):
    # Se obtienen las correlaciones simultáneas
    # Se eliminan los nan
    qq2 = ~(np.isnan(EPTriM[:,i])|np.isnan(EqTriM[:,i]))
    ECorrPvq[i,i] = st.pearsonr(EPTriM[qq2,i],EqTriM[qq2,i])[0]
    ESigPvq[i,i] = st.pearsonr(EPTriM[qq2,i],EqTriM[qq2,i])[1] 
    for j in range(4):
        if j > i:
            qq2 = ~(np.isnan(EPTriM[:,j])|np.isnan(EqTriM[:,i]))
            ECorrPvq[j,i] = st.pearsonr(EPTriM[qq2,j],EqTriM[qq2,i])[0]
            ESigPvq[j,i] = st.pearsonr(EPTriM[qq2,j],EqTriM[qq2,i])[1]
        if j < i:
            qq2 = ~(np.isnan(EPTriM[1:,j])|np.isnan(EqTriM[:-1,i]))
            ECorrPvq[j,i] = st.pearsonr(EPTriM[1:,j][qq2],EqTriM[:-1,i][qq2])[0]
            ESigPvq[j,i] = st.pearsonr(EPTriM[1:,j][qq2],EqTriM[:-1,i][qq2])[1]


# Se arma la matriz de correlación trimestrales de P con Uq EST
ECorrPvUq = np.zeros((4,4))
ESigPvUq = np.zeros((4,4))

for i in range(4):
    # Se obtienen las correlaciones simultáneas
    # Se eliminan los nan
    qq3 = ~(np.isnan(EPTriM[:,i])|np.isnan(EUqTriM[:,i]))
    ECorrPvUq[i,i] = st.pearsonr(EPTriM[qq3,i],EUqTriM[qq3,i])[0]
    ESigPvUq[i,i] = st.pearsonr(EPTriM[qq3,i],EUqTriM[qq3,i])[1] 
    for j in range(4):
        if j > i:
            qq3 = ~(np.isnan(EPTriM[:,j])|np.isnan(EUqTriM[:,i]))
            ECorrPvUq[j,i] = st.pearsonr(EPTriM[qq3,j],EUqTriM[qq3,i])[0]
            ESigPvUq[j,i] = st.pearsonr(EPTriM[qq3,j],EUqTriM[qq3,i])[1]
        if j < i:
            qq3 = ~(np.isnan(EPTriM[1:,j])|np.isnan(EUqTriM[:-1,i]))
            ECorrPvUq[j,i] = st.pearsonr(EPTriM[1:,j][qq3],EUqTriM[:-1,i][qq3])[0]
            ESigPvUq[j,i] = st.pearsonr(EPTriM[1:,j][qq3],EUqTriM[:-1,i][qq3])[1]



     
        
# Se obtienen los promedios trimestrales PM
# Se obtienen los promedios trimestrales
PTri = [np.nanmean(PM[:2])]
UTri = [np.nanmean(U_prom[:2])]
qTri = [np.nanmean(q_prom[:2])]
UqTri = [np.nanmean(Uxq_prom[:2])]
        
for i in range(2,PM.shape[0]-1,3):
    print(i+1)
    PTri.append(np.nanmean(PM[i:i+3]))
    UTri.append(np.nanmean(U_prom[i:i+3]))
    qTri.append(np.nanmean(q_prom[i:i+3]))
    UqTri.append(np.nanmean(Uxq_prom[i:i+3]))
    
PTriM = np.reshape(PTri,(-1,4))
UTriM = np.reshape(UTri,(-1,4))
qTriM = np.reshape(qTri,(-1,4))
UqTriM = np.reshape(UqTri,(-1,4))


# Se arma la matriz de correlación trimestrales de P con U
CorrPvU = np.zeros((4,4))
SigPvU = np.zeros((4,4))

for i in range(4):
    # Se obtienen las correlaciones simultáneas
    # Se eliminan los nan
    qq = ~(np.isnan(PTriM[:,i])|np.isnan(UTriM[:,i]))
    CorrPvU[i,i] = st.pearsonr(PTriM[qq,i],UTriM[qq,i])[0]
    SigPvU[i,i] = st.pearsonr(PTriM[qq,i],UTriM[qq,i])[1] 
    for j in range(4):
        if j > i:
            qq = ~(np.isnan(PTriM[:,j])|np.isnan(UTriM[:,i]))
            CorrPvU[j,i] = st.pearsonr(PTriM[qq,j],UTriM[qq,i])[0]
            SigPvU[j,i] = st.pearsonr(PTriM[qq,j],UTriM[qq,i])[1]
        if j < i:
            qq = ~(np.isnan(PTriM[1:,j])|np.isnan(UTriM[:-1,i]))
            CorrPvU[j,i] = st.pearsonr(PTriM[1:,j][qq],UTriM[:-1,i][qq])[0]
            SigPvU[j,i] = st.pearsonr(PTriM[1:,j][qq],UTriM[:-1,i][qq])[1]


# Se arma la matriz de correlación trimestrales de P con q
CorrPvq = np.zeros((4,4))
SigPvq = np.zeros((4,4))

for i in range(4):
    # Se obtienen las correlaciones simultáneas
    # Se eliminan los nan
    qq2 = ~(np.isnan(PTriM[:,i])|np.isnan(qTriM[:,i]))
    CorrPvq[i,i] = st.pearsonr(PTriM[qq2,i],qTriM[qq2,i])[0]
    SigPvq[i,i] = st.pearsonr(PTriM[qq2,i],qTriM[qq2,i])[1] 
    for j in range(4):
        if j > i:
            qq2 = ~(np.isnan(PTriM[:,j])|np.isnan(qTriM[:,i]))
            CorrPvq[j,i] = st.pearsonr(PTriM[qq2,j],qTriM[qq2,i])[0]
            SigPvq[j,i] = st.pearsonr(PTriM[qq2,j],qTriM[qq2,i])[1]
        if j < i:
            qq2 = ~(np.isnan(PTriM[1:,j])|np.isnan(qTriM[:-1,i]))
            CorrPvq[j,i] = st.pearsonr(PTriM[1:,j][qq2],qTriM[:-1,i][qq2])[0]
            SigPvq[j,i] = st.pearsonr(PTriM[1:,j][qq2],qTriM[:-1,i][qq2])[1]


# Se arma la matriz de correlación trimestrales de P con Uq
CorrPvUq = np.zeros((4,4))
SigPvUq = np.zeros((4,4))

for i in range(4):
    # Se obtienen las correlaciones simultáneas
    # Se eliminan los nan
    qq3 = ~(np.isnan(PTriM[:,i])|np.isnan(UqTriM[:,i]))
    CorrPvUq[i,i] = st.pearsonr(PTriM[qq3,i],UqTriM[qq3,i])[0]
    SigPvUq[i,i] = st.pearsonr(PTriM[qq3,i],UqTriM[qq3,i])[1] 
    for j in range(4):
        if j > i:
            qq3 = ~(np.isnan(PTriM[:,j])|np.isnan(UqTriM[:,i]))
            CorrPvUq[j,i] = st.pearsonr(PTriM[qq3,j],UqTriM[qq3,i])[0]
            SigPvUq[j,i] = st.pearsonr(PTriM[qq3,j],UqTriM[qq3,i])[1]
        if j < i:
            qq3 = ~(np.isnan(PTriM[1:,j])|np.isnan(UqTriM[:-1,i]))
            CorrPvUq[j,i] = st.pearsonr(PTriM[1:,j][qq3],UqTriM[:-1,i][qq3])[0]
            SigPvUq[j,i] = st.pearsonr(PTriM[1:,j][qq3],UqTriM[:-1,i][qq3])[1]













# Se eliminan valores nan de la serie de precipitación

qq=~np.isnan(PM)

#Correlaciones datos crudos

CorrPM_q=st.pearsonr(PM[qq],q_prom[qq])

CorrPM_U=st.pearsonr(PM[qq],U_prom[qq])
CorrPM_Uxq=st.pearsonr(PM[qq],Uxq_prom[qq])

# Correlaciones rezagadas PM con U
qq = ~(np.isnan(PM) | np.isnan(U_prom))
Psinnan = PM[qq]
Usinnan = U_prom[qq]
CorrPrecRez_U = [st.pearsonr(Psinnan,Usinnan)[0]]
SigPrecRez_U = [st.pearsonr(Psinnan,Usinnan)[1]]
for i in range(1,6):
    # Se eliminan los datos faltantes
    Unan = U_prom[:-i]
    Pnan = PM[i:]
    qq = ~(np.isnan(Pnan) | np.isnan(Unan))
    Psinnan = Pnan[qq]
    Usinnan = Unan[qq]
    
    CorrPM_U = st.pearsonr(Psinnan,Usinnan)
    CorrPrecRez_U.append(CorrPM_U[0])
    SigPrecRez_U.append(CorrPM_U[1])

# Correlaciones rezagadas PM con q
qq = ~(np.isnan(PM) | np.isnan(q_prom))
Psinnan = PM[qq]
qsinnan = q_prom[qq]
CorrPrecRez_q = [st.pearsonr(Psinnan,qsinnan)[0]]
SigPrecRez_q = [st.pearsonr(Psinnan,qsinnan)[1]]
for i in range(1,6):
    # Se eliminan los datos faltantes
    qnan = q_prom[:-i]
    Pnan = PM[i:]
    qq = ~(np.isnan(Pnan) | np.isnan(qnan))
    Psinnan = Pnan[qq]
    qsinnan = qnan[qq]
    
    CorrPM_q = st.pearsonr(Psinnan,qsinnan)
    CorrPrecRez_q.append(CorrPM_q[0])
    SigPrecRez_q.append(CorrPM_q[1])
    
    
# Correlaciones rezagadas PM con Uq
qq = ~(np.isnan(PM) | np.isnan(Uxq_prom))
Psinnan = PM[qq]
Uqsinnan = Uxq_prom[qq]
CorrPrecRez_Uq = [st.pearsonr(Psinnan,Uqsinnan)[0]]
SigPrecRez_Uq = [st.pearsonr(Psinnan,Uqsinnan)[1]]
for i in range(1,6):
    # Se eliminan los datos faltantes
    Uqnan = Uxq_prom[:-i]
    Pnan = PM[i:]
    qq = ~(np.isnan(Pnan) | np.isnan(Uqnan))
    Psinnan = Pnan[qq]
    Uqsinnan = Uqnan[qq]
    
    CorrPM_Uq = st.pearsonr(Psinnan,Uqsinnan)
    CorrPrecRez_Uq.append(CorrPM_Uq[0])
    SigPrecRez_Uq.append(CorrPM_Uq[1])    
    

  
    
#plt.title(r'Hola $f=2+2 \alpha \tau \beta$')
    
    
#precipitacion rezagada respecto a viento zonal
f = plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size':32})
plt.bar(np.arange(-0.4,5,1),CorrPrecRez_U,width=0.8)
plt.title('Precipitación vs Viento Zonal', fontsize=38)
plt.xlabel(r'$ \tau$',fontsize=30)
plt.ylabel(r'$ \rho$',fontsize=30)
plt.grid()
 



#precipitacion rezagada respecto a humedad específica
f = plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size':32})
plt.bar(np.arange(-0.4,5,1),CorrPrecRez_q,width=0.8)
plt.title('Precipitación vs Humedad Específica', fontsize=38)
plt.xlabel(r'$ \tau$',fontsize=30)
plt.ylabel(r'$ \rho$',fontsize=30)
plt.grid()
 

#precipitacion rezagada respecto a U x humedad específica
f = plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size':32})
plt.bar(np.arange(-0.4,5,1),CorrPrecRez_Uq,width=0.8)
plt.title('Precipitación vs Transporte de Humedad', fontsize=38)
plt.xlabel(r'$ \tau$',fontsize=30)
plt.ylabel(r'$ \rho$',fontsize=30)
plt.grid()
 




PEst=np.array(PEst)
UEst=np.array(UEst)
qEst=np.array(qEst)
UqEst=np.array(UqEst)
 




#Correlaciones datos estandarizados

ECorrPM_q=st.pearsonr(PEst[qq],qEst[qq])

ECorrPM_U=st.pearsonr(PEst[qq],UEst[qq])
ECorrPM_Uxq=st.pearsonr(PEst[qq],UqEst[qq])

# Correlaciones rezagadas PM con U
qq = ~(np.isnan(PEst) | np.isnan(UEst))
EPsinnan = PEst[qq]
EUsinnan = UEst[qq]
ECorrPrecRez_U = [st.pearsonr(EPsinnan,EUsinnan)[0]]
ESigPrecRez_U = [st.pearsonr(EPsinnan,EUsinnan)[1]]
for i in range(1,6):
    # Se eliminan los datos faltantes
    EUnan = UEst[:-i]
    EPnan = PEst[i:]
    qq = ~(np.isnan(EPnan) | np.isnan(EUnan))
    EPsinnan = EPnan[qq]
    EUsinnan = EUnan[qq]
    
    ECorrPM_U = st.pearsonr(EPsinnan,EUsinnan)
    ECorrPrecRez_U.append(ECorrPM_U[0])
    ESigPrecRez_U.append(ECorrPM_U[1])

# Correlaciones rezagadas PM con q

qq = ~(np.isnan(PEst) | np.isnan(qEst))
EPsinnan = PEst[qq]
Eqsinnan = qEst[qq]
ECorrPrecRez_q = [st.pearsonr(EPsinnan,Eqsinnan)[0]]
ESigPrecRez_q = [st.pearsonr(EPsinnan,Eqsinnan)[1]]
for i in range(1,6):
    # Se eliminan los datos faltantes
    Eqnan = qEst[:-i]
    EPnan = PEst[i:]
    qq = ~(np.isnan(EPnan) | np.isnan(Eqnan))
    EPsinnan = EPnan[qq]
    Eqsinnan = Eqnan[qq]
    
    ECorrPM_q = st.pearsonr(EPsinnan,Eqsinnan)
    ECorrPrecRez_q.append(ECorrPM_q[0])
    ESigPrecRez_q.append(ECorrPM_q[1])    
    
# Correlaciones rezagadas PM con Uq

qq = ~(np.isnan(PEst) | np.isnan(UqEst))
EPsinnan = PEst[qq]
EUqsinnan = UqEst[qq]
ECorrPrecRez_Uq = [st.pearsonr(EPsinnan,EUqsinnan)[0]]
ESigPrecRez_Uq = [st.pearsonr(EPsinnan,EUqsinnan)[1]]
for i in range(1,6):
    # Se eliminan los datos faltantes
    EUqnan = UqEst[:-i]
    EPnan = PEst[i:]
    qq = ~(np.isnan(EPnan) | np.isnan(EUqnan))
    EPsinnan = EPnan[qq]
    EUqsinnan = EUqnan[qq]
    
    ECorrPM_Uq = st.pearsonr(EPsinnan,EUqsinnan)
    ECorrPrecRez_Uq.append(ECorrPM_Uq[0])
    ESigPrecRez_Uq.append(ECorrPM_Uq[1])

  
#graficos

#precipitacion rezagada respecto a viento zonal EST
f = plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size':32})
plt.bar(np.arange(-0.4,5,1),ECorrPrecRez_U,color='k',width=0.8)
plt.title('Precipitación vs Viento Zonal', fontsize=38)
plt.xlabel(r'$ \tau$',fontsize=30)
plt.ylabel(r'$ \rho$',fontsize=30)
plt.grid()
 
#precipitacion rezagada respecto a humedad específica EST
f = plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size':32})
plt.bar(np.arange(-0.4,5,1),ECorrPrecRez_q,color='k',width=0.8)
plt.title('Precipitación vs Humedad Específica', fontsize=38)
plt.xlabel(r'$ \tau$',fontsize=30)
plt.ylabel(r'$ \rho$',fontsize=30)
plt.grid()
 
#precipitacion rezagada respecto a U x humedad específica EST
f = plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size':32})
plt.bar(np.arange(-0.4,5,1),ECorrPrecRez_Uq,color='k',width=0.8)
plt.title('Precipitación vs Transporte de Humedad', fontsize=38)
plt.xlabel(r'$ \tau$',fontsize=30)
plt.ylabel(r'$ \rho$',fontsize=30)
plt.grid()


#GRAFICOS





zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz



#Viento Zonal


Fig= plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size':14})
plt.plot(U_prom[:],'-k',label='vel.viento')
plt.title('Velocidad del viento', fontsize=24)
plt.xlabel('Fechas(meses)',fontsize=18)
plt.ylabel('Velocidad del viento[m/s]',fontsize=18)
plt.legend(loc=0)
#plt.savefig(Pathimg+'Vel_viento_Prom_1'+'.png')
#plt.close('all')

#Humedad Especif{ica}


#F = plt.figure()
#A1 = plt.plot(PM)
#plt.ylabel('Precipitación [mm]')
#A11 = plt.twinx()
#A11.plot(PEst,'r--',lw=2.5)
#A11.set_ylabel('Precipitación Estandarizada')


zzzzzzzzzzzzzzzzzzz

#se guara el doc

Nameout= Pathout+'Series de tiempo6.xlsx'

#se abre el cod

W=xlsxwl.Workbook(Nameout)

# Se crea la hoja

WS= W.add_worksheet('series')

# se copian los encabezados

WS.write(0,0,'Fechas')
WS.write(0,1,'Precipitación')
WS.write(0,2,'Viento Zonal')
WS.write(0,3,'Humedad Específica')
WS.write(0,4,'Viento Zonal x Humedad Específica')
WS.write(0,5,'z_Precipitación')
WS.write(0,6,'z_Viento Zonal')
WS.write(0,7,'z_Humedad Específica')
WS.write(0,8,'z_Viento Zonal x Humedad Específica')



FechaExcel = [i.strftime('%Y'+'/'+'%m'+'/'+'%d') for i in FechaMen]

x = 1 # Contador de filas de Excel
for i in range(len(PM)):
    WS.write(x,0,FechaExcel[i])
    if np.isnan(PM[i]):
        WS.write(x,1,'')
    else:
        WS.write(x,1,PM[i])
    WS.write(x,2,U_prom[i])
    WS.write(x,3,q_prom[i])
    WS.write(x,4,Uxq_prom[i])
    WS.write(x,6,UEst[i])
    WS.write(x,7,qEst[i]) 
    WS.write(x,8,UqEst[i]) 
    x += 1
    
    
W.close()



#se guara el doc

Nameout= Pathout+'Series de tiempo6.xlsx'

#se abre el cod

W=xlsxwl.Workbook(Nameout)

# Se crea la hoja

WS= W.add_worksheet('series')

# se copian los encabezados

WS.write(0,0,'Fechas')
WS.write(0,1,'z_Precipitación')

x = 1 # Contador de filas de Excel
for i in range(len(PEst)):
    WS.write(x,0,FechaExcel[i])
    if np.isnan(PEst[i]):
        WS.write(x,1,'')
  
    else:
      
        WS.write(x,1,PEst[i])
        
        WS.write(x,2,UEst[i])
        x += 1    

    
W.close()


# Se grafican los datos
#plt.plot(FechaRean,U_prom)


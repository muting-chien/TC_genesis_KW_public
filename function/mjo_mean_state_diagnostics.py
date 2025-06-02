# -*- coding: utf-8 -*-
"""
MJO_Mean state diagnostics
Created on Thu Jun 25 10:26:07 2020

@author: USER
"""

'''
Load Packages that are necessary for the analysis
'''
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from matplotlib.cm import get_cmap
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import scipy.signal as signal
from scipy import interpolate
from scipy.signal import butter, filtfilt


'''
0. Technical problems:
'''
def irange(start, stop, step): #inclusive_range
    return range(start, (stop + 1) if step >= 0 else (stop - 1), step)

'''
1. Basics: remove nan, calculate anomaly, transform time format(yyyymmdd to year_month_day), 
           select seasons, meridional average
'''
def inpaint_nans(x,critical_val,big_or_small): #assume x is 3-dim, assume nan is the same in time
    if big_or_small == 1: # >critical value-->nan
        i = np.squeeze( np.argwhere(x[1,:,:] > critical_val) )
        #print(i)
        print(np.shape(i))
    elif big_or_small == 0: # <critical value-->nan
        i = np.squeeze( np.argwhere(x[1,:,:] < critical_val) )
    elif big_or_small == -1:
        i = np.squeeze( np.argwhere( np.isnan(x)==1 ))
        print(i)
        print(np.shape(i))
    if np.sum(i) == 0:
        print('no nan data')
        x_nonan = x
    else:
        print('has nan data!!!')
        nansize = np.size(i,0)
        a = np.empty([np.size(x,0),nansize])
        a[:] = np.nan
        x_nonan = x
        #x_nonan[i[:,0],i[:,1],i[:,2]] = a    
        x_nonan[:,i[:,0],i[:,1]] = a  
        
        for j in range(0,1): #nansize):
            #j0 = i[j,0]
            j1 = i[j,0] #[j,1]
            j2 = i[j,1] #[j,2]
            
            '''
            if j0 ==0:
                J0L = np.nan
            else:
                J0L = x_nonan[j0-1,j1,j2]
                
            if j0 == np.size(x_nonan,0)-1:
                J0R = np.nan
            else:
                J0R = x_nonan[j0+1,j1,j2]
            '''   
            if j1 == 0:
                J1L = np.empty([np.size(x,0)])
                J1L[:] = np.nan
            else:
                #J1L = x_nonan[j0,j1-1,j2]
                J1L = x_nonan[:,j1-1,j2]
                
            if j1 == np.size(x_nonan,1)-1:
                J1R = np.empty([np.size(x,0)])
                J1R[:] = np.nan
            else:
                #J1R = x_nonan[j0,j1+1,j2]
                J1R = x_nonan[:,j1+1,j2]
                
            if j2 == 0:
                J2L = np.empty([np.size(x,0)])
                J2L[:] = np.nan
            else:
                #J2L = x_nonan[j0,j1,j2-1]
                J2L = x_nonan[:,j1,j2-1]
                
            if j2 == np.size(x_nonan,2)-1:
                J2R = np.nan([np.size(x,0)])
            else:
                J2R = np.empty([np.size(x,0)])
                J2R[:] = np.nan              
            #x_nonan[j0,j1,j2] = np.nanmean(np.array([J0L,J0R,J1L,J1R,J2L,J2R]))   
            #for k in range(0,np.size(x,0)):
            x_nonan[:,j1,j2] = np.nanmean(np.array([J1L,J1R,J2L,J2R]),0)
            print(J1L)
            print(J1R)
            print(J2L)
            print(J2R)
            print(x_nonan[:,j1,j2])
    return x_nonan

def inpaint_nans_2(x,critical_val,big_or_small): #assume x is 3-dim, assume nan is the same in time dimension
    if big_or_small == 1: # >critical value-->nan
        x = np.where(x>critical_val,np.nan,x)
    elif big_or_small == 0: # <critical value-->nan
        x = np.where(x<critical_val,np.nan,x)
        
    i = np.argwhere(np.isnan(x[1,:,:])==1)
    #print(np.shape(i))
    print('portion of nan data:'+str(np.size(i,0)/((np.size(x,1)*np.size(x,2)))) )
    if np.sum(np.shape(i)) == 0:
        print('no nan data')
        x_nonan = x
    else:
        print('has nan data!!!')
        nansize = np.size(i,0)
        x_nonan = x      

        for j in range(0,nansize):
            j1 = i[j,0] #[j,1]
            j2 = i[j,1] #[j,2]
            if j1 == 0:
                J1L = np.empty([np.size(x,0)])
                J1L[:] = np.nan                
            else:
                J1L = x_nonan[:,j1-1,j2]

            if j1 == np.size(x_nonan,1)-1:
                J1R = np.empty([np.size(x,0)])
                J1R[:] = np.nan
            else:
                J1R = x_nonan[:,j1+1,j2]

            if j2 == 0:
                J2L = np.empty([np.size(x,0)])
                J2L[:] = np.nan
            else:
                J2L = x_nonan[:,j1,j2-1]

            if j2 == np.size(x_nonan,2)-1:
                J2R = np.empty([np.size(x,0)])
                J2R[:] = np.nan
            else:
                J2R = x_nonan[:,j1,j2+1]             
            x_nonan[:,j1,j2] = np.nanmean(np.array([J1L,J1R,J2L,J2R]),0)
    print(np.argwhere(np.isnan(x_nonan)==1))
    return x_nonan

def inpaint_nans_3(x,critical_val,big_or_small): #assume x is 3-dim, assume nan is the same in time dimension
    if big_or_small == 1: # >critical value-->nan
        x = np.where(x>critical_val,np.nan,x)
    elif big_or_small == 0: # <critical value-->nan
        x = np.where(x<critical_val,np.nan,x)
        
    x_nonan = x
    if np.sum(np.shape(np.argwhere(np.isnan(x)==1)))== 0:
        print('no nan data') 
    else:
        print('has nan data!!!')    
        for t in range(0,np.size(x,0)):
            i = np.argwhere(np.isnan(x[t,:,:])==1)
            #print(np.shape(i))
            if t == 0:
                print('portion of nan data:'+str(np.size(i,0)/((np.size(x,1)*np.size(x,2)))) )

            nansize = np.size(i,0)     

            for j in range(0,nansize):
                j1 = i[j,0] #[j,1]
                j2 = i[j,1] #[j,2]
                if j1 == 0: 
                    #J1L = np.empty([np.size(x,0)])
                    J1L = np.nan                
                else:
                    J1L = x_nonan[t,j1-1,j2]

                if j1 == np.size(x,1)-1:
                    #J1R = np.empty([np.size(x,0)])
                    J1R = np.nan
                else:
                    J1R = x_nonan[t,j1+1,j2]

                if j2 == 0:
                    #J2L = np.empty([np.size(x,0)])
                    J2L = np.nan
                else:
                    J2L = x_nonan[t,j1,j2-1]

                if j2 == np.size(x,2)-1:
                    #J2R = np.empty([np.size(x,0)])
                    J2R = np.nan
                else:
                    J2R = x_nonan[t,j1,j2+1]             
                x_nonan[t,j1,j2] = np.nanmean(np.array([J1L,J1R,J2L,J2R]))
                if np.isnan(x_nonan[t,j1,j2])==1:
                    print('t='+str(t))
                    print('j1='+str(j1))
                    print('j2='+str(j2))
            if len(np.argwhere(np.isnan(x_nonan)==1))==0:
                print('succesfully inpaint nan')
    #print(np.argwhere(np.isnan(x_nonan)==1))
    return x_nonan

def filled_to_nan(x,critical_val,big_or_small): 
    
    if big_or_small == 1: # >critical value-->nan
        x_nan = np.where(x>critical_val,np.nan,x)
    elif big_or_small == 0: # <critical value-->nan            
        x_nan = np.where(x<critical_val,np.nan,x)
    
    if np.sum(np.isnan(x_nan))== 0:
        print('no nan data') 
    else:
        print('has nan data!!!')   
        
    return x_nan


def remove_anncycle(x): #x is one dimension
    mmax = 7 #(7-1)/2= 3; remove mean and first 3 harmonics
    n = np.size(x)
    #t = np.arange(0,n,1)
    t = np.arange(1,n+1,1)
    x = np.matrix(x).T
    A = np.zeros([mmax,n])
    for m in range(1,mmax+1):
        if m == 1:
            A[m-1,:] = np.ones([1,n])
        elif np.mod(m,2) == 1:
            A[m-1,:] = np.ones([1,n])*np.sin((m-1)/2*2*np.pi/365*t)
        else:
            A[m-1,:] = np.ones([1,n])*np.cos(m/2*2*np.pi/365*t)
    C= np.matmul(A,x).T/n
    xcycle = np.matmul(C,A).T
    xp = x-xcycle
    return xp, xcycle

def remove_anncycle_2d(x,time,lon):
    # caution: before using remove_anncycle function, make sure there is no nan in the data, otherwise it won't work
    x_ano = np.zeros([np.size(time),np.size(lon)])
    x_cyc = np.zeros([np.size(time),np.size(lon)])
    
    #remove annual cycle
    for ilon in range(0,np.size(lon,0)):
            x_a, x_c=  remove_anncycle( np.squeeze(x[:,ilon]) )         
            x_ano[:,ilon] = np.squeeze(x_a)
            x_cyc[:,ilon] = np.squeeze(x_c)         
    #print("finish removing anncycle")
    return x_ano, x_cyc


def remove_anncycle_3d(x,time,lat,lon):
    # caution: before using remove_anncycle function, make sure there is no nan in the data, otherwise it won't work
    x_ano = np.zeros([np.size(time),np.size(lat),np.size(lon)])
    x_cyc = np.zeros([np.size(time),np.size(lat),np.size(lon)])
    
    #remove annual cycle
    for ilat in range(0,np.size(lat,0)):
        for ilon in range(0,np.size(lon,0)):
            x_a, x_c=  remove_anncycle( np.squeeze(x[:,ilat,ilon]) )         
            x_ano[:,ilat,ilon] = np.squeeze(x_a)
            x_cyc[:,ilat,ilon] = np.squeeze(x_c)         
    #print("finish removing anncycle")
    return x_ano, x_cyc

def remove_anncycle_4d(x,time,lev,lat,lon):
    # caution: before using remove_anncycle function, make sure there is no nan in the data, otherwise it won't work
    x_ano = np.zeros([np.size(time),np.size(lev),np.size(lat),np.size(lon)])
    x_cyc = np.zeros([np.size(time),np.size(lev),np.size(lat),np.size(lon)])
    
    #remove annual cycle
    for ilev in range(0,np.size(lev,0)):
        for ilat in range(0,np.size(lat,0)):
            for ilon in range(0,np.size(lon,0)):
                x_a, x_c=  remove_anncycle( np.squeeze(x[:,ilev,ilat,ilon]) )       
                x_ano[:,ilev,ilat,ilon] = np.squeeze(x_a)
                x_cyc[:,ilev,ilat,ilon] = np.squeeze(x_c)         
    return x_ano, x_cyc

def mean_var(x,x_ano,x_f): # caution: assume time in the 0th direction
    # Calculate mean of original data, variance of anomaly/filtered data
    #if dim == 3:
    x_m = np.nanmean(x,0) # mean
    #if dim == 4:
        #x_m = np.mean(x[:,ilev,:,:],0) # mean
    x_av = np.nanvar(x_ano,0) #variance of anomaly    
    x_fv = np.nanvar(x_f,0) #variance of intraseasonal signal
    return x_m, x_av, x_fv

def yyyymmdd_y_m_d(dates): #transfer yyyymmdd into year mon day (3 matrix)
    year = np.zeros(np.size(dates))
    mon = np.zeros(np.size(dates))
    day = np.zeros(np.size(dates))
    for i in range(0,np.size(dates)):
        time_str = str(dates[i]) #original format of time is yyyymmdd
        year[i] = int(time_str[0:4])
        mon[i] = int(time_str[4:6])
        day[i] = int(time_str[6:8])   
    return year,mon,day
    
#dates = time
def Seasonal_Selection(x,dates,MonID,dimt):
    # dimt: time in which dimension of the data x 
    dimx = np.size(np.shape(x)) #x is 1d,2d,3d,4d     
    year,mon,day = yyyymmdd_y_m_d(dates)
    #ff = []
    #ff = None
    if MonID == 'ANN':
        ff = np.arange(0,np.size(dates))
    elif MonID == 'DJF':
        ff = np.argwhere( (mon==12)|(mon==1)|(mon==2) )
    elif MonID == 'MAM':
        ff = np.argwhere( (mon==3)|(mon==4)|(mon==5) ) 
    elif MonID == 'JJA':
        ff = np.argwhere( (mon==6)|(mon==7)|(mon==8) )
    elif MonID == 'SON':
        ff = np.argwhere( (mon==9)|(mon==10)|(mon==11) )
    else:
        for i in range(0,12):
            if MonID == str(i+1):
                ff = np.argwhere(mon==i+1)
                break        
            
    if dimx == 1 and dimt == 0:
        x_season = x[ff]          
    elif dimx == 2 and dimt == 0:
        x_season = x[ff,:]
    elif dimx == 2 and dimt == 1:
        x_season = x[:,ff]
    elif dimx == 3 and dimt == 0:
        x_season = x[ff,:,:] 
    elif dimx == 3 and dimt == 1:
        x_season = x[:,ff,:]         
    elif dimx == 3 and dimt == 2:
        x_season = x[:,:,ff]           
    elif dimx == 4 and dimt == 0:
        x_season = x[ff,:,:,:]
    elif dimx == 4 and dimt == 1:
        x_season = x[:,ff,:,:] 
    elif dimx == 4 and dimt == 2:
        x_season = x[:,:,ff,:]        
    elif dimx == 4 and dimt == 3:
        x_season = x[:,:,:,ff]    
    x_season = np.squeeze(x_season)
    date_season = dates[ff]
    return x_season,date_season

def mer_ave_4d(x, lat, latdim): #assume x is (time, plev, lat, lon)
    cos_lat = np.cos( np.deg2rad(lat) )
    if np.sum(np.isnan(x)==1)==0: # no nan
        x_mer_ave = np.average(x, latdim, weights=cos_lat)
    else: # is nan
        print('has nan, but use nanmean')
        nt = np.size(x,0)
        nlev = np.size(x,1)
        nlat = np.size(x,2)
        nlon = np.size(x,3)
        x_mer_ave = np.empty((nt,nlev,nlon))
        x_mer_ave[:] = np.nan
        for it in range(0,nt):
            for ilev in range(0,nlev):
                for ilon in range(0,nlon):
                    x2 = x[it,ilev,:,ilon]
                    indices = ~np.isnan(x2)
                    x_mer_ave[it,ilev,ilon] = np.average(x2[indices], weights=cos_lat[indices])
    return x_mer_ave



def mer_ave(x,lat,latdim): #assume x is (time,lat,lon)
    cos_lat = np.cos( np.deg2rad(lat) )
    if np.sum(np.isnan(x)==1)==0: #no nan
        x_mer_ave = np.average(x, latdim, weights=cos_lat)
    else: #is nan
        print('has nan, but use nanmean')
        nt = np.size(x,0)
        nlon = np.size(x,2)
        nlat = np.size(x,1)
        x_mer_ave = np.empty((nt,nlon)) 
        x_mer_ave[:] = np.nan
        for it in range(0,nt):
            for ilon in range(0,nlon):
                x2 = x[it,:,ilon]
                indices = ~np.isnan(x2)
                x_mer_ave[it,ilon] = np.average(x2[indices], weights=cos_lat[indices])
    return x_mer_ave

def mer_ave_2d(x,lat): #assume x is (lat,lon)
    cos_lat = np.cos( np.deg2rad(lat) )
    if np.sum(np.isnan(x)==1)==0:
        x_mer_ave = np.average(x, 0, weights=cos_lat)
    else:
        print('has nan, but use nanmean')
        nlon = np.size(x,1)
        nlat = np.size(x,0)
        if np.size(np.shape(x))==3:
            nt = np.size(x,2)
            x_mer_ave = np.empty((nlon,nt))
        else:
            nt = 1
            x_mer_ave = np.empty((nlon)) 
        x_mer_ave[:] = np.nan
        for it in range(0,nt):
            for ilon in range(0,nlon):
                if nt !=1:
                    x2 = x[:,ilon,it]
                else:
                    x2 = x[:,ilon]
                indices = ~np.isnan(x2)
                if nt == 1:
                    x_mer_ave[ilon] = np.average(x2[indices], weights=cos_lat[indices])
                else:
                    x_mer_ave[ilon,it] = np.average(x2[indices], weights=cos_lat[indices])
    return x_mer_ave

'''
Filter function

(1) butter_bandpass: band
input:
- lowcut: frequency of low limit of the band
- highcut: frequency of high limit of the band
- fs: frequency of the data
- order: order of the filter

(2) butter_bandpass_filter: filtered data
input:
- data: target data to be filtered
- lowcut: frequency of low limit of the band
- highcut: frequency of high limit of the band
- fs: frequency of the data
- order: order of the filter
output:
- y: filtered data

# Sample:
# 10-day high pass filter 
t = np.arange(0,100)
a = np.sin(2*np.pi*t/5)+np.sin(2*np.pi*t/20) #original data with 5 day and 20 day signal
af = MJO.butter_highpass_filter(a, 1/10, 1) 
    # 1/10-->10 day high pass
    # 1-->1 data point is 1 day
# af will only retain 20 day signal

# Caution: 
# the filtered dimension should be the last dimension of the input data
# ex: in the above sameple, time should be the last dimension of a 
'''
#def autocorr(x, t=1):
    #return np.corrcoef(np.array([x[:-t], x[t:]]))

from scipy.signal import butter, freqz, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(cut, fs, order=4):
    nyq = 0.5 * fs
    cut = cut / nyq
    b, a = butter(order, cut, btype='highpass')
    return b, a

def butter_highpass_filter(data, cut, fs, order=5):
    b, a = butter_highpass(cut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass(cut, fs, order=4):
    nyq = 0.5 * fs
    cut = cut / nyq
    b, a = butter(order, cut, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cut, fs, order=5):
    b, a = butter_lowpass(cut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

'''
Applying filtering
'''
def remove_10d_from_3hr_data(x, fs=8, cutoff=1/10):
    time_dim = 0
    # fs = 8-->8 samples per day (3-hourly)
    # cutoff = 1/10 (1/day)-->cut off signals with period longer than 10 day
    dimsize = np.size(np.shape(x))
    nt = np.size(x,0)
    if dimsize == 4:
        n1 = np.size(x,1)
        n2 = np.size(x,2)
        n3 = np.size(x,3)
        for i1 in range(0,n1):
            for i2 in range(0,n2):
                for i3 in range(0,n3):
                    if i1 == 0 and i2 == 0 and i3 == 0:
                        x_m = np.empty([nt,n1,n2,n3])
                    x_m[:,i1,i2,i3] = butter_lowpass_filter(x[:,i1,i2,i3], cutoff, fs)
 
    elif dimsize == 3:
        n1 = np.size(x,1)
        n2 = np.size(x,2)
        for i1 in range(0,n1):
            for i2 in range(0,n2):
                if i1 == 0 and i2 == 0:
    	            x_m = np.empty([nt,n1,n2]) 
                x_m[:,i1,i2] = butter_lowpass_filter(x[:,i1,i2], cutoff, fs)
    elif dimsize == 2:
        n1 = np.size(x,1)
        for i1 in range(0,n1):
            if i1 == 0:
                x_m = np.empty([nt,n1])
            x_m[:,i1] = butter_lowpass_filter(x[:,i1], cutoff, fs)
    elif dimsize == 1:
        x_m = butter_lowpass_filter(x, cutoff, fs)

    xano = x-x_m
    del x_m
    return xano



'''
EOF function
'''
def normalize_before_ceof(u850_f_merave,u200_f_merave,olr_f_merave): #(time,lon)
    # normalize the data before doing eof, 
    # you need to do this because the unit of the two dataset is not the same
    u850_f_merave = np.transpose(u850_f_merave)
    u200_f_merave = np.transpose(u200_f_merave)
    olr_f_merave  = np.transpose(olr_f_merave)
    nlon = np.size(u850_f_merave,0)
    nt = np.size(u850_f_merave,1)
    
    mu_u850 = np.nanmean(u850_f_merave)
    std_u850 = np.nanstd(u850_f_merave)
    u850_norm = ( u850_f_merave - mu_u850 )/std_u850
    
    mu_u200 = np.nanmean(u200_f_merave)
    std_u200 = np.nanstd(u200_f_merave)
    u200_norm = ( u200_f_merave - mu_u200 )/std_u200
    
    mu_olr = np.nanmean(olr_f_merave) 
    std_olr = np.nanstd(olr_f_merave)
    olr_norm = ( olr_f_merave - mu_olr )/std_olr

    X = np.zeros([nlon*3,nt])
    X[0:nlon,:] = u850_norm
    X[nlon:2*nlon,:] = u200_norm
    X[2*nlon:3*nlon,:] = olr_norm
    return X, mu_u850, std_u850,\
           mu_u200, std_u200,\
           mu_olr,  std_olr  

def eof(xx): #x=x(structure dim, sampling dim)
    u, s, v = np.linalg.svd(xx, full_matrices=False)
    EOF = np.transpose(u) #EOFi=EOF[i-1] EOF1=EOF[0],EOF2=EOF[1],....
    #print('u:',np.shape(u),' ,xx:',np.shape(xx))
    PC = np.dot(np.transpose(u),xx) #PCi =pc[i-1], pc1=pc[0],pc2=pc[1],...
    nt = np.size(xx,1)
    eigval = s**2/nt
    eigval_explained_var = eigval/np.sum(eigval)*100 #percent
    
    # calculate degree of freedom so that we can do North test
    L = 1 #one-lag auto-corelation
    B = 0
    for k in range(L-1,nt-L):
        B = B + np.sum( xx[:,k]*xx[:,k+L] )
    
    phi_L = 1/(nt-2*L)*B
    phi_0 = 1/nt*np.sum( xx**2 )
    r_L     = phi_L/phi_0
    #r_L     = np.nanmean(phi_L)/np.nanmean(phi_0)
    dof = ( 1-r_L**2 )/( 1+r_L**2 )*nt
    
    eigval_err = eigval_explained_var*np.sqrt(2/dof)
    return EOF, PC, eigval, eigval_explained_var, eigval_err, dof, phi_0, phi_L
    
def rmm_eight_phase_index(rmm1,rmm2,time_f,rmm1_ann,rmm2_ann):
    rmm1_norm = (rmm1-np.mean(rmm1_ann))/np.std(rmm1_ann) #normalized rmm1
    rmm2_norm = (rmm2-np.mean(rmm2_ann))/np.std(rmm2_ann) #normalized rmm2
    n = np.zeros(9)
    RMM_ind = np.empty((9,np.size(time_f)))
    RMM_ind[:] = np.NaN
    for i in range(0,np.size(time_f)):
        n = n.astype(int)
        RMM1 = rmm1_norm[i]
        RMM2 = rmm2_norm[i]
        A = RMM1**2+RMM2**2
        if (A<1).any(): #weak MJO
            RMM_ind[8,n[8]]=i
            n[8]=n[8]+1
        elif RMM1<0 and RMM2<0 and np.abs(RMM1)>np.abs(RMM2): #PHASE1
            RMM_ind[0,n[0]]=i
            n[0]=n[0]+1
        elif RMM1<0 and RMM2<0 and np.abs(RMM1)<np.abs(RMM2): #2
            RMM_ind[1,n[1]]=i
            n[1]=n[1]+1
        elif RMM1>0 and RMM2<0 and np.abs(RMM1)<np.abs(RMM2): #3 
            RMM_ind[2,n[2]]=i
            n[2]=n[2]+1
        elif RMM1>0 and RMM2<0 and np.abs(RMM1)>np.abs(RMM2): #4
            RMM_ind[3,n[3]]=i
            n[3]=n[3]+1
        elif RMM1>0 and RMM2>0 and np.abs(RMM1)>np.abs(RMM2): #5
            RMM_ind[4,n[4]]=i
            n[4]=n[4]+1
        elif RMM1>0 and RMM2>0 and np.abs(RMM1)<np.abs(RMM2): #6
            RMM_ind[5,n[5]]=i
            n[5]=n[5]+1
        elif RMM1<0 and RMM2>0 and np.abs(RMM1)<np.abs(RMM2): #7
            RMM_ind[6,n[6]]=i
            n[6]=n[6]+1
        elif RMM1<0 and RMM2>0 and np.abs(RMM1)>np.abs(RMM2): #8
            RMM_ind[7,n[7]]=i
            n[7]=n[7]+1       
    return n,RMM_ind,rmm1_norm,rmm2_norm

def eight_phase_composite(x_f,RMM_ind):
    x_f_8ph = np.zeros([8,np.size(x_f,1),np.size(x_f,2)])
    for ph in range(0,8):
        i = np.squeeze( np.argwhere(~np.isnan(RMM_ind[ph,:])) ) 
        ii = RMM_ind[ph,i]
        ii = ii.astype(int)   
        x_f_8ph[ph,:,:] = np.mean(x_f[ii,:,:],0)
    return x_f_8ph   

def plot_rmm_8phase_composite(v_in,vname,lon,lat,clev,cticks,fig_dire,model_name,CMAP):
    monid = list(['DJF','MAM','JJA','SON'])#,'NDJFMA','MJJASO'])
    lon2d,lat2d = np.meshgrid(lon,lat)
    for ss in range(0,4):#np.size(monid)):  
        fig,ax = plt.subplots(8,figsize=(12, 9))
        plt.rcParams.update({'font.size': 13.5})
        fig_name = model_name+'8phases_'+vname+'_'+monid[ss]+'.png'
        for ph in range(0,8):        
            temp = np.squeeze(v_in[ss,ph,:,:])
            ax[ph] = plt.subplot(8,1,ph+1)
            if ph == 0:
                plt.title(model_name+' '+monid[ss]+' Phase1(top)-8(bottom) composites: '+vname, fontsize=13.5)
            m = Basemap(projection='merc',llcrnrlat=-20,urcrnrlat=20,\
                            llcrnrlon=50,urcrnrlon=230,lat_ts=20,resolution='c') #50E-230E   
            #m = Basemap(projection='merc',llcrnrlat=-20,urcrnrlat=20,\
            #                llcrnrlon=lon[0],urcrnrlon=lon[-1],lat_ts=20,resolution='c') #50E-230E
            x, y = m(to_np(lon2d), to_np(lat2d))
            olr_contourf = m.contourf(x, y, temp, levels = clev, cmap=get_cmap(CMAP), zorder=2, alpha=0.99,extend = 'both')
            if ph == 7:
                m.drawmeridians(np.arange(0, 360, 90),labels=[False, False, False, True])
            else:
                m.drawmeridians(np.arange(0, 360, 90),labels=[False, False, False, False])          
            m.drawcoastlines(color='black')
            m.drawmeridians(np.arange(0, 360, 90),labels=[False, False, False, False])
            m.drawparallels(np.arange(-20, 30, 10),labels=[False, True, False, False])
            m.drawmapboundary(fill_color='white')
            del(temp)   
        olr_cb = fig.colorbar(olr_contourf,ax=ax[0:8],orientation = 'horizontal',shrink=.6,\
                             fraction=0.046, pad=0.04)
        olr_cb.ax.tick_params(labelsize=12)
        olr_cb.set_ticks(cticks)    
        fig.savefig(fig_dire+fig_name,format='png', dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()        

        
'''
Moisture Budget
'''
def z_integrate(x,plev):
    g = 9.8 #m/s^2
    nt   = np.size(x,0)
    nlev = np.size(x,1)
    nlat = np.size(x,2)
    nlon = np.size(x,3)
    Pmat = np.tile(plev,[nlat, nlon,1]) #equivalent to matlab: repmat
    Pmat = np.transpose(Pmat, (2, 0, 1)) # equivalent to matlab: Pmat = permute(Pmat,[2,3,1]);
    dp = np.zeros([nlev-1, nlat ,nlon])
    for ii in range(0,nlev-1):
        dp[ii,:,:] = Pmat[ii+1,:,:]-Pmat[ii,:,:]
    
    x_mid = np.zeros([nt,nlev-1,nlat,nlon])
    for ii in range(0,nlev-1): #lev
        x_mid[:,ii,:,:] = (x[:,ii+1,:,:]+x[:,ii,:,:])/2 # Takes the midpoint of each pressure level
    
    # Vertical Integration
    xdP = np.zeros([nt,nlev-1,nlat,nlon])
    for tt in range(0,nt):
        temp1 = x_mid[tt,:,:,:]
        xdP[tt,:,:,:] = temp1*dp  # Pa
        del temp1    
    Col_x = (1/g)*np.squeeze(np.sum(xdP,1))
    if dp[0,0,0]<0:
        Col_x = -Col_x

    print('finish vertical integration')
    return Col_x

def z_integrate_2d(x,plev): #(t,lev)
    g = 9.8 #m/s^2
    nt   = np.size(x,0)
    nlev = np.size(x,1)
    Pmat = plev
    dp = np.zeros([nlev-1])
    for ii in range(0,nlev-1):
        dp[ii] = Pmat[ii+1]-Pmat[ii]

    x_mid = np.zeros([nt,nlev-1])
    for ii in range(0,nlev-1): #lev
        x_mid[:,ii] = (x[:,ii+1]+x[:,ii])/2 # Takes the midpoint of each pressure level

    # Vertical Integration
    xdP = np.zeros([nt,nlev-1])
    for tt in range(0,nt):
        temp1 = x_mid[tt,:]
        xdP[tt,:] = temp1*dp  # Pa
        del temp1
    Col_x = (1/g)*np.squeeze(np.sum(xdP,1))
    if dp[0]<0:
        Col_x = -Col_x
    
    print('finish vertical integration')
    return Col_x

def z_integrate_1d(x,plev): #(lev)
    g = 9.8 #m/s^2
    nlev = np.size(x,0)
    Pmat = plev
    dp = np.zeros([nlev-1])
    for ii in range(0,nlev-1):
        dp[ii] = Pmat[ii+1]-Pmat[ii]

    x_mid = np.zeros([nlev-1])
    for ii in range(0,nlev-1): #lev
        x_mid[ii] = (x[ii+1]+x[ii])/2 # Takes the midpoint of each pressure level

    # Vertical Integration
    xdP = np.zeros([nlev-1])
    xdP = x_mid*dp  # Pa
    del x_mid

    Col_x = (1/g)*np.squeeze(np.sum(xdP,0))
    print('test')
    if dp[0]<0:
        print('change sign')
        Col_x = -Col_x

    print('finish vertical integration')
    return Col_x

    
def z_integrate_bpfilter(x,plev,nt,nlat,nlon,do_midpoint_pressure):
    # this is used for moisture budget calculation of each term
    # input data is x(time,lev,lat,lon) = dqdt udqdx vdqdy wdqdp
    # if you have already done midpoint pressure average of the variable before using this function
    #   do_midpoint_pressure=0 -->  wdqdP
    # if you have not,
    #   do_midpoint_pressure=1 --> dqdt,udqdx,vdqdy
    
    g = 9.8 #m/s^2
    nlev = np.size(plev)
    Pmat = np.tile(plev,[nlat-2, nlon, 1]) # equivalent to matlab: repmat
    Pmat = np.transpose(Pmat, (2, 0, 1)) # equivalent to matlab: Pmat = permute(Pmat,[2,3,1]);
    dp = np.zeros([nlev-1, nlat-2 ,nlon])
    for ii in range(0,nlev-1):
        dp[ii,:,:] = Pmat[ii+1,:,:]-Pmat[ii,:,:]
    
    if np.size(x,2) != nlat-2:
        x = x[:,:,1:-1,:] #remove lat = latmax and latmin
        
    if np.size(x,0) != nt-2:
        x = x[1:-1,:,:,:] #remove 1st and last day

    #print(np.shape(x))
    x_mid = np.zeros([nt-2,nlev-1,nlat-2,nlon])
    if do_midpoint_pressure == 1:
        
        for ii in range(0,nlev-1): #lev
            x_mid[:,ii,:,:] = (x[:,ii+1,:,:]+x[:,ii,:,:])/2 # Takes the midpoint of each pressure level
            #print(ii)
    else:
        x_mid = x
    
    # Vertical Integration
    xdP = np.zeros([nt-2,nlev-1,nlat-2,nlon])
    for tt in range(0,nt-2):
        temp1 = x_mid[tt,:,:,:]
        xdP[tt,:,:,:] = np.squeeze(temp1*dp)  # Pa
        del temp1    
    Col_x = (1/g)*np.squeeze(np.sum(xdP,1))

    if dp[0,0,0]<0:
        Col_x = -Col_x
 
    print('finish vertical integration')
    
    # Remove mean and seasonal cycles
    Col_x_a = np.zeros([nt-2,nlat-2,nlon])
    for ilat in range(0,nlat-2):
        for ilon in range(0,nlon):
            x_a, x_c=  remove_anncycle( Col_x[:,ilat,ilon]  )          
            Col_x_a[:,ilat,ilon] = np.squeeze(x_a)
            del x_a, x_c  
    
    # Band pass filtering 
    Col_x_BP = np.zeros([nt-2,nlat-2,nlon])
    T_high = 20 #day
    T_low = 100 #day
    T = 1 #1 grid point = T day (time resolution)
    for ilat in range(0,np.size(Col_x,1)):
        for ilon in range(0,np.size(Col_x,2)):           #select seasons, meridional average
            Col_x_BP[:,ilat,ilon]=\
            butter_bandpass_filter(Col_x_a[:,ilat,ilon],1/T_low,1/T_high,1/T,order=4)
    del xdP, x, x_mid, Col_x_a
    Col_x_BP = Col_x_BP[364:-1-363,:,:] #remove 1st year and last year
    return Col_x_BP


'''
Moisture Budget Projection
'''


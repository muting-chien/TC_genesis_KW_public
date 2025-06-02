####################################
# Goal of this code is to do Fourier wavenumber-frequency filtering of variables
# Input: Anomaly (time, lat, lon) or KW-meridional-projected anomalies (time, lon)  
# 2023.7.30
# Mu-Ting Chien
#################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import scipy.signal as signal
import scipy.stats as stat
from tkinter import Tcl
from netCDF4 import Dataset
import statsmodels.api as sm # comment this line if executing .py file instead of .ipynb
import glob 
import math
import cmaps
#
import sys
sys.path.append('/glade/work/muting/function/')
import mjo_mean_state_diagnostics as MJO
import create_my_colormap as mycolor
RWB = mycolor.red_white_blue()
#import tool as tool #This only works if the script is .py, not .ipynb

##############################################
# Part 1: Geneal tool, Non KW specific
##############################################

##########################################################
def find_critical_coh2(dof):
    # This function find the critical coh^2 value given the 0.01 statistical significance level and degree of freedom
    
    dof_list = np.array([2,3,4,5,6,7,8,9,10,\
                11,12,13,14,15,16,17,18,19,20,\
                25,30,35,40,45,\
                50,60,70,80,90,100,\
                125,150,175,200])
    
    coh_crit_1percent_list = np.array([0.99,0.901,0.785,0.684,0.602,0.536,0.482,0.438,0.401,\
                              0.37,0.342,0.319,0.298,0.28,0.264,0.25,0.237,0.226,0.215,\
                              0.175,0.147,0.127,0.112,0.1,\
                              0.09,0.075,0.065,0.057,0.052,0.045,\
                              0.036,0.031,0.026,0.023])
    
    #np.savez('/glade/work/muting/function/coh2_statistical_test.npz',dof=dof, coh_crit_1percent=coh_crit_1percent)
    
    idof = np.argwhere(dof_list==dof).squeeze()
    if np.size(idof)!=0:
        coh2_crit = coh_crit_1percent_list[idof]
    else:
        ddof = np.abs(dof_list-dof)
        idof = np.argwhere(ddof == np.min(ddof)).squeeze()
        if dof > dof_list[idof]:
            idofL = idof
            idofR = idof+1
        else:
            idofL = idof-1
            idofR = idof
        
        # lINEAR INTERPOLATION
        fracL = (dof-dof_list[idofL])/(dof_list[idofR]-dof_list[idofL])
        coh2_crit = fracL*coh_crit_1percent_list[idofR] + (1-fracL)*coh_crit_1percent_list[idofL] # 
    
    return coh2_crit


###################################
def nan_to_value_by_interp(V):
    
    # V is designed to be 4 dimension (time, plev, lat, lon), replace nan by the mean of the average of the nearby values
    
    V_nonan = np.where( np.isnan(V)==1, np.nan, V)
    
    nanloc = np.argwhere(np.isnan(V))
    nannum = np.size(nanloc,0)
    vdim   = np.size(nanloc,1)
    
    if vdim >= 1:
        nt = np.size(V,0) # typically nt
    if vdim >=2:
        nlev = np.size(V,1) # typically nlev
    if vdim >=3:
        nlat = np.size(V,2) # typically nlat
    if vdim >=4:
        nlon = np.size(V,3) # typically nlon

    
    for i in range(0,nannum):
        
        # Time dimension
        if nanloc[i,0] == 0:
            V_previous = np.nan
        else:
            V_previous = V[nanloc[i,0]-1, nanloc[i,1], nanloc[i,2], nanloc[i,3]]

        if nanloc[i,0] == nt-1:
            V_later = np.nan
        else:
            V_later = V[nanloc[i,0]+1, nanloc[i,1], nanloc[i,2], nanloc[i,3]]
        
        # Plev dimension
        if nanloc[i,1] == 0:
            V_up = np.nan
        else:
            V_up = V[nanloc[i,0], nanloc[i,1]-1, nanloc[i,2], nanloc[i,3]]
        
        if nanloc[i,1] == nlev-1:
            V_down = np.nan
        else:
            V_down = V[nanloc[i,0], nanloc[i,1]+1, nanloc[i,2], nanloc[i,3]] 
            
        # lat dimension
        if nanloc[i,2] == nlat-1:
            V_N = np.nan
        else:
            V_N = V[nanloc[i,0], nanloc[i,1], nanloc[i,2]+1, nanloc[i,3]]

        if nanloc[i,2] == 0:
            V_S = np.nan
        else:
            V_S = V[nanloc[i,0], nanloc[i,1], nanloc[i,2]-1, nanloc[i,3]]
        
        # Lon dimension
        if nanloc[i,3] == nlon-1:
            V_E = V[nanloc[i,0], nanloc[i,1], nanloc[i,2], 0]
        else:
            V_E = V[nanloc[i,0], nanloc[i,1], nanloc[i,2], nanloc[i,3]+1]

        if nanloc[i,3] == 0:
            V_W = V[nanloc[i,0], nanloc[i,1], nanloc[i,2], -1]
        else:
            V_W = V[nanloc[i,0], nanloc[i,1], nanloc[i,2], nanloc[i,3]-1]        
    
        if V_E == np.nan and V_W == np.nan and V_N == np.nan and V_S==np.nan and V_previous==np.nan and V_later==np.nan and V_down == np.nan:
            print('Caution: nan is directly replaced by the value at the upper pressure level ')
    
        V_nonan[nanloc[i,0], nanloc[i,1], nanloc[i,2], nanloc[i,3]] = np.nanmean( np.array([V_E, V_W, V_N, V_S, V_previous, V_later, V_up, V_down]))
    
    return V_nonan


##########################################################
def smoothing_121(V, npass=10):
    # Assume 1-2-1 smoothing dimension is the 0th dimension, V can be 1D, 2D, 3D, npass is how many passes of 1-2-1 filters are used
    Vdim = np.size(np.shape(V)) # 1D or 2D or 3D
    
    if Vdim == 1:
        n0 = np.size(V,0)
        V_smooth       = np.empty([n0])
        V_smooth[0]    = V[0]
        V_smooth[-1]   = V[-1]
        V_smooth[1:-1] = V[1:-1]
        
        for i in range(0,npass):
            V_smooth[1:-1] = 1/4*V_smooth[:-2] + 1/2*V_smooth[1:-1] + 1/4*V_smooth[2:]
    
    elif Vdim == 2: 
        n0 = np.size(V,0)
        n1 = np.size(V,1)
        V_smooth       = np.empty([n0,n1])
        V_smooth[0,:]    = V[0,:]
        V_smooth[-1,:]   = V[-1,:]
        V_smooth[1:-1,:] = V[1:-1,:]
        
        for i in range(0,npass):
            V_smooth[1:-1,:] = 1/4*V_smooth[:-2,:] + 1/2*V_smooth[1:-1,:] + 1/4*V_smooth[2:,:]
    
    elif Vdim == 3: 
        n0 = np.size(V,0)
        n1 = np.size(V,1)
        n2 = np.size(V,2)
        V_smooth       = np.empty([n0,n1,n2])
        V_smooth[0,:,:]    = V[0,:,:]
        V_smooth[-1,:,:]   = V[-1,:,:]
        V_smooth[1:-1,:,:] = V[1:-1,:,:]
        
        for i in range(0,npass):
            V_smooth[1:-1,:,:] = 1/4*V_smooth[:-2,:,:] + 1/2*V_smooth[1:-1,:,:] + 1/4*V_smooth[2:,:,:]
    
    return V_smooth

##########################################################################





###################################################
def find_nfile_skip(CASENAME, CASENAME_SHORT, iyr_min, iyr_max):
    # Goal of this is to find how many files are missing in work_output_pressure_coord_full 
    # so that we know what is the maximum size of time for variables. 
    # This output will be used in load_2D_data_as_1variable
    
    # CASENAME = SST_AQP3_Qobs_27_3h_20y
    # CASENAME_SHORT = CTL_20y_3h_20y is the initial characters of the name in work_output_pressure_coord_full
    
    # Initialization:
    nfile_skip = 0
    
    file_dir = '/glade/derecho/scratch/muting/FROM_CHEYENNE/work_output_pressure_coord_full/'+CASENAME+'/'
    for iyr in range(iyr_min, iyr_max+1):
        yyyy = "%04d" %(iyr+1) 
        for imon in range(0,12):
            mm = "%02d" %(imon+1) 
            for ifile in range(0,8):
                dd = "%02d" %(ifile)
                file_name = CASENAME_SHORT+'_'+yyyy+mm+dd+'.nc'
                f = glob.glob(file_dir+file_name)
                if len(f)==0:
                    nfile_skip = nfile_skip+1
                    continue
                
    return nfile_skip

#######################################################
def load_2D_data_as_1variable(CASENAME,vname1, vname2, iyr_min, iyr_max, latmax, org2post_const, nfile_skip, KW_proj):
    # vname1 is "precc"
    # vname2 is "precl" or "", if vname2 ="", only vname1 will be loaded
    # iyr_min = yr to start (02-->year 3)
    # iyr_max = yr to end (04-->year 5)
    # latmax = 15 for the tropics
    # org2post_const = constant that convert original raw data to 1 single variable with our desired unit. 
    #   For example, original unit of precip is kg/m^2/s. To convert to mm/day, we need to multiply by 1000*s2d.
    # KW_proj = 1: Do meridional projection
    # KW_proj = 0: Do not do meridional projection
    
    # Constants
    nmon = 12
    ist   = 0
    nyr = iyr_max-iyr_min+1
    nt = (nyr*nmon*8-nfile_skip)*31
    
    # load CESM 3-hourly output
    for iyr in range(iyr_min,iyr_max+1):
        yr_str = "%04d" %(iyr+1)
        #print('YEAR:',yr_str)
        for imon in range(0,nmon):
            mon_str = "%02d" %(imon+1)
            # Load 3-hourly data
            file_dir = '/glade/derecho/scratch/muting/FROM_CHEYENNE/archive/'+CASENAME+'/atm/hist/'
            file_in  = file_dir+CASENAME+'.cam.h1.'+yr_str+'-'+mon_str+'*.nc'
            f        = glob.glob(file_in)
            if len(f)==0:
                continue
            f3 = Tcl().call('lsort', '-dict', f) #sort the file name by date (small to large)
                    
            for iday in range(0,len(f)):
                day_str = "%02d" %(iday)
                data    = Dataset( f3[iday], "r", format="NETCDF4")
                if  iday==0 and imon == 0 and iyr == iyr_min:
                    lat = data.variables['lat'][:]
                    lon = data.variables['lon'][:]
                    nlat = np.size(lat)
                    nlon = np.size(lon)
                    # trimm narrower tropical band for space-time spectrum calculation
                    latmin = -latmax
                    dmin = np.abs(lat-latmin)
                    dmax = np.abs(lat-latmax)
                    imin = np.argwhere(dmin==np.min(dmin)).squeeze()
                    imax = np.argwhere(dmax==np.min(dmax)).squeeze()
                    lat_tropics = lat[imin:imax+1]
                    nlat_tropics = np.size(lat_tropics)
                    time = np.arange(0,nt)
                    # Initialization
                    if KW_proj == 1:
                        V = np.empty([nt,nlon])
                    else:
                        V = np.empty([nt,nlat_tropics,nlon])
                        
                temp1 = data.variables[vname1][:,imin:imax+1,:]*org2post_const #m/s-->mm/day #convective precip
                if KW_proj == 1:
                    temp1 = KW_meridional_projection(temp1, lat_tropics, 0)
                nt_small = np.size(temp1,0)
                if nt_small!=31:
                    print('Skip this file, size of time not 31, YEAR:',yr_str,', MON:',mon_str,', FILE:',day_str)
                    continue
                if len(vname2)!=0: # WITH 2ND VARIABLE
                    temp2 = data.variables[vname2][:,imin:imax+1,:]*org2post_const
                    
                    if KW_proj == 1:
                        temp2 = KW_meridional_projection(temp2, lat_tropics, 0)
                        V[ist:ist+nt_small,:] = temp1+temp2
                    else:
                        V[ist:ist+nt_small,:,:]  = temp1+temp2
                        
                    del temp2
                else: # NO 2ND VARIABLE
                    
                    if KW_proj == 1:
                        V[ist:ist+nt_small,:] = temp1
                    else:
                        V[ist:ist+nt_small,:,:]  = temp1
                del temp1
                ist = ist+nt_small 
    
    if KW_proj == 1:
        V = V[:ist,:]
    else:
        V = V[:ist,:,:]
    time = time[:ist]
                
                        
    return V, time, lat_tropics, lon

#################################################
# Updated
def load_3D_data_as_1variable(CASENAME, CASENAME_SHORT, vname, iyr_min, iyr_max, latmax, nfile_skip, kw_proj=1):
    # vname is list(['vname1','vname2','vname3']), number of element in the list could be variable, up to 4 variables
    # iyr_min = yr to start (02-->year 3)
    # iyr_max = yr to end (04-->year 5)
    # latmax = 10 for the tropics
    #   For example, original unit of precip is kg/m^2/s. To convert to mm/day, we need to multiply by 1000*s2d.
    # Default is KW projection = 1

    # Constants
    nmon = 12
    nt_s = 31
    ist   = 0
    nyr = iyr_max-iyr_min+1
    nt_m = nyr*nmon*8-nfile_skip # to calculate mean state
    nt = nt_m*nt_s
    nfile_skip_test = 0

    # Find how many variables
    vnum = np.size(vname)

    DIR_in = '/glade/derecho/scratch/muting/FROM_CHEYENNE/'
    file_dir = DIR_in+'work_output_pressure_coord_full/'+CASENAME+'/'
    ###################
    # load CESM 3-hourly output!
    for iyr in range(iyr_min,iyr_max+1):
        yr_str = "%04d" %(iyr+1)
        for imon in range(0,nmon):
            mon_str = "%02d" %(imon+1)
            for iday in range(0,8):
                day_str = "%02d" %(iday)
                file_in = file_dir+CASENAME_SHORT+'_'+yr_str+mon_str+day_str+'.nc'
                f = glob.glob(file_in)
                if len(f)==0:
                    nfile_skip_test = nfile_skip_test+1
                    continue
                data    = Dataset( file_in, "r", format="NETCDF4")
                if  iday == 0 and imon == 0 and iyr==iyr_min:
                    lon = data.variables['lon'][:]
                    nlon = np.size(lon)
                    lat_org = data.variables['lat'][:]
                    dmin = np.abs(lat_org+latmax)
                    dmax = np.abs(lat_org-latmax)
                    imin = np.argwhere(dmin==np.min(dmin)).squeeze()
                    imax = np.argwhere(dmax==np.min(dmax)).squeeze()
                    lat = lat_org[imin:imax+1]
                    nlat = np.size(lat)
                    #
                    plev = data.variables['plev'][:]
                    nlev = np.size(plev)
                    #
                    time = np.arange(0,nt)
                    if vnum>=1:
                        V1  = np.empty([nt,nlev,nlat,nlon])
                        V1m = np.zeros([nlev])
                    if vnum>=2:
                        V2 = np.empty([nt,nlev,nlat,nlon])
                        V2m = np.zeros([nlev])
                    if vnum>=3:
                        V3 = np.empty([nt,nlev,nlat,nlon])
                        V3m = np.zeros([nlev])
                    if vnum==4:
                        V4 = np.empty([nt,nlev,nlat,nlon])
                        V4m = np.zeros([nlev])

                if vnum>=1:
                    temp= data.variables[vname[0]][:,imin:imax+1,:,:]
                    nt_small = np.size(temp,0)

                if nt_small != nt_s:
                    print('Skip with this file, size of time not 31, YEAR:',yr_str,', MON:',mon_str,', FILE:',day_str)
                    continue

                if vnum>=1:
                    V1m  = V1m + MJO.mer_ave_2d( np.nanmean(np.nanmean(temp,2),0), lat)
                    temp = np.transpose(temp, (0,3,1,2))
                    V1[ist:ist+nt_s,:,:,:] = temp
                if vnum>=2:
                    temp = data.variables[vname[1]][:,imin:imax+1,:,:]
                    V2m  = V2m + MJO.mer_ave_2d( np.nanmean(np.nanmean(temp,2),0), lat)
                    temp = np.transpose(temp, (0,3,1,2))
                    V2[ist:ist+nt_s,:,:,:] = temp
                if vnum>=3:
                    temp = data.variables[vname[2]][:,imin:imax+1,:,:]
                    V3m  = V3m + MJO.mer_ave_2d( np.nanmean(np.nanmean(temp,2),0), lat)
                    temp = np.transpose(temp, (0,3,1,2))
                    V3[ist:ist+nt_s,:,:,:] = temp
                if vnum==4:
                    temp = data.variables[vname[3]][:,imin:imax+1,:,:]
                    V4m  = V4m + MJO.mer_ave_2d( np.nanmean(np.nanmean(temp,2),0), lat)
                    temp = np.transpose(temp, (0,3,1,2))
                    V4[ist:ist+nt_s,:,:,:] = temp

                del data

                ist = ist+nt_s

    time = np.arange(0,ist)

    # Calculate mean
    if vnum == 4:
        V4 = V4[:ist,:,:,:]
        #V4 = nan_to_value_by_interp(V4)
        if kw_proj == 1:
            V4 = KW_meridional_projection(V4, lat, 0)
        V4m_final = V4m/nt_m
    if vnum >= 3:
        V3 = V3[:ist,:,:,:]
        #V3 = nan_to_value_by_interp(V3)
        if kw_proj == 1:
            V3 = KW_meridional_projection(V3, lat, 0)
        V3m_final = V3m/nt_m
    if vnum >= 2:
        V2 = V2[:ist,:,:,:]
        #V2 = nan_to_value_by_interp(V2)
        if kw_proj == 1:
            V2 = KW_meridional_projection(V2, lat, 0)
        V2m_final = V2m/nt_m
    if vnum >= 1:
        V1 = V1[:ist,:,:,:]
        #V1 = nan_to_value_by_interp(V1), this is not needed as we will do meridional projection which ignore nan later
        if kw_proj == 1:
            V1 = KW_meridional_projection(V1, lat, 0)
        V1m_final = V1m/nt_m
    
        
    if kw_proj == 1:
        if vnum == 1:
            return V1, V1m_final, time, plev, lon
        if vnum == 2:
            return V1, V2, V1m_final, V2m_final, time, plev, lon
        if vnum == 3:
            return V1, V2, V3, V1m_final, V2m_final, V3m_final, time, plev, lon
        if vnum == 4:
            return V1, V2, V3, V4, V1m_final, V2m_final, V3m_final, V4m_final, time, plev, lon
    else:
        if vnum == 1:
            return V1, V1m_final, time, plev, lon, lat
        if vnum == 2:
            return V1, V2, V1m_final, V2m_final, time, plev, lon, lat
        if vnum == 3:
            return V1, V2, V3, V1m_final, V2m_final, V3m_final, time, plev, lon, lat
        if vnum == 4:
            return V1, V2, V3, V4, V1m_final, V2m_final, V3m_final, V4m_final, time, plev, lon, lat        


########################################
def z_integrate_3d(x,plev): #(t,lev,lon)
    # Note that plev should be in Pa
    
    g = 9.8 #m/s^2
    nt   = np.size(x,0)
    nlev = np.size(x,1)
    nlon = np.size(x,2)
    Pmat = np.tile(plev,[nlon,1]) #equivalent to matlab: repmat
    Pmat = np.transpose(Pmat, (1, 0)) # equivalent to matlab: Pmat = permute(Pmat,[2,3,1]);
    dp = np.zeros([nlev-1, nlon])

    for ii in range(0,nlev-1):
        dp[ii,:] = Pmat[ii+1,:]-Pmat[ii,:]

    x_mid = np.zeros([nt,nlev-1,nlon])
    for ii in range(0,nlev-1): #lev
        x_mid[:,ii,:] = (x[:,ii+1,:]+x[:,ii,:])/2 # Takes the midpoint of each pressure level

    # Vertical Integration
    xdP = np.zeros([nt,nlev-1,nlon])
    for tt in range(0,nt):
        temp1 = x_mid[tt,:,:]
        xdP[tt,:,:] = temp1*dp  # Pa
        del temp1
    Col_x = (1/g)*np.squeeze(np.sum(xdP,1))
    if dp[0,0]<0:
        Col_x = -Col_x
        
    return Col_x

######################################
def cc_equation(T, plev): # T in K
    
    # Define constant
    epsilon = 0.622
    
    # Check dimension
    if np.size(np.shape(T))==3: # If 3D variable
        nt = np.size(T,0)
        nlon = np.size(T,2)
        plev_large = np.tile(plev, (nt,nlon,1))
        plev_large = np.transpose(plev_large, (0,2,1))
    elif np.size(np.shape(T))==2: # If 2D variable
        nt = np.size(T,0)
        plev_large = np.tile(plev, (nt, 1))
    elif np.size(np.shape(T))==1: # If 1D variable
        plev_large = plev
    
    e = 6.1094*np.exp( 17.625*(T-273.15)/(T-273.15+243.04) ) # hPa
    qvs = epsilon*e/plev_large
    
    return e, qvs

#############################################
def calc_saturation_fraction(T, qv, plev):
    
    # Calculate saturated vapor pressure 
    e, qvs = cc_equation(T, plev)  #CC equation, hPa
    del e
    
    # Calculate saturation fraction
    col_qv  = z_integrate_3d(qv, plev*100)
    col_qvs = z_integrate_3d(qvs, plev*100)
    sf = col_qv/col_qvs # fraction
    
    return sf, col_qv, col_qvs


###############################################
def calc_mse(T, z, qv, plev, saturation):
    
    # Define constant
    Cpd = 1004
    g = 9.8
    Lv = 2.25*10**6
    epsilon = 0.622
    
    # Check dimension
    if np.size(np.shape(T))==3: # If 3D variable
        nt = np.size(T,0)
        nlon = np.size(T,2)
        plev_large = np.tile(plev, (nt,nlon,1))
        plev_large = np.transpose(plev_large, (0,2,1))
    elif np.size(np.shape(T))==2: # If 2D variable
        nt = np.size(T,0)
        plev_large = np.tile(plev, (nt, 1))
    elif np.size(np.shape(T))==1: # If 1D variable
        plev_large = plev
    
    
    if saturation == 0:
        MSE = Cpd*T+g*z+Lv*qv
    elif saturation == 1:
        e = 6.1094*np.exp( 17.625*(T-273.15)/(T-273.15+243.04) ) #CC equation, hPa
        qvs = epsilon*e/plev_large
        MSE = Cpd*T+g*z+Lv*qvs
    
    return MSE


################################                        
# Calcualte Sp, N^2
def calculate_Sp_N2_Hs(Tm, Zm, plev):
    
    # Define constants
    g  = 9.8
    Rd = 287
    Cp = 1004
    gamma_d = g/Cp
    
    Hs = Rd*np.nanmean(Tm)/g
    gamma = -(Tm[2:]-Tm[:-2])/(Zm[2:]-Zm[:-2]) #(nlev-2)
    Sp = (gamma_d-gamma)*Rd*Tm[1:-1]/(plev[1:-1]*100)/g   
    N2 = Rd/Hs*(gamma_d-gamma)
    plev_new = plev[1:-1]
    
    return Sp, N2, Hs, plev_new



######################################################
def calculate_power_spectrum(V, kw_meridional_proj, Fs_t=8, Fs_lon=1/2.5,  output_sym_only=1): # V(time, lon) or V(time, lat, lon)
    # kw_meridional_proj = 0 or 1

    # subset into segments in time (96 days, overlap 60 days)
    nt = np.size(V,0)
    seglen  = int(96*Fs_t)  
    overlap = int(60*Fs_t)  
    Hw = int(5*Fs_t)   #width of Hann window
    n = int(seglen-overlap)#average seglen (not counting the overlap part)
    nseg = math.floor((nt-seglen)/n)+1
    dof = 2*nseg

    if kw_meridional_proj == 0:
            
        # Separate into symmetric/antisymmetric component
        nlat = np.size(V,1)
        nlon = np.size(V,2)
        nlat_half = int((nlat+1)/2) #include equator
        V_sym = np.zeros([nt,nlat_half,nlon]) 
        V_asy = np.zeros([nt,nlat_half,nlon]) 
        for ilat in range(0,nlat_half):
            V_sym[:,ilat,:] = (V[:,ilat,:]+V[:,nlat-ilat-1,:])/2
            V_asy[:,ilat,:] = -(V[:,ilat,:]-V[:,nlat-ilat-1,:])/2
        
        # Subset into segments in time
        V_sym_seg = np.zeros([nseg,seglen,nlat_half,nlon])
        V_asy_seg = np.zeros([nseg,seglen,nlat_half,nlon])
        HANN = np.concatenate((np.hanning(Hw),np.ones(seglen-Hw*2),np.hanning(Hw)),axis=0)
        HANN = np.tile(HANN,(nlon,nlat_half,1))
        HANN = HANN.transpose(2, 1, 0)
        for iseg in range(0,nseg):
            iseg_n = int(iseg*n)
            V_sym_seg[iseg,:,:,:] = signal.detrend(V_sym[iseg*n:iseg*n+seglen,:,:],axis=0)*HANN
            V_asy_seg[iseg,:,:,:] = signal.detrend(V_asy[iseg*n:iseg*n+seglen,:,:],axis=0)*HANN
        del V_sym, V_asy

        # calculate space-time spectrum
        FFT_V_sym = np.zeros([nseg,seglen,nlat_half,nlon],dtype=complex)
        FFT_V_asy = np.zeros([nseg,seglen,nlat_half,nlon],dtype=complex)
        for ilat in range(0,nlat_half):
            for iseg in range(0,nseg):
                FFT_V_sym[iseg,:,ilat,:] = np.fft.fft2(V_sym_seg[iseg,:,ilat,:])/(nlon*seglen)*4
                FFT_V_asy[iseg,:,ilat,:] = np.fft.fft2(V_asy_seg[iseg,:,ilat,:])/(nlon*seglen)*4
        del V_sym_seg, V_asy_seg

        # Power spectrum
        C_sym = FFT_V_sym*np.conj(FFT_V_sym) # Power spectrum of V
        Cm_sym = np.nanmean(np.nanmean(C_sym,2),0)
        C_asy = FFT_V_asy*np.conj(FFT_V_asy) # Power spectrum of V
        Cm_asy = np.nanmean(np.nanmean(C_asy,2),0)
            
    else:
        nlon  = np.size(V,1)
        V_sym_seg = np.zeros([nseg,seglen,nlon])

        HANN = np.concatenate((np.hanning(Hw),np.ones(seglen-Hw*2),np.hanning(Hw)),axis=0)
        HANN = np.tile(HANN,(nlon,1))
        HANN = HANN.transpose(1, 0)
        for iseg in range(0,nseg):
            iseg_n = int(iseg*n)
            V_sym_seg[iseg,:,:] = signal.detrend(V[iseg*n:iseg*n+seglen,:],axis=0)*HANN
            
        # calculate space-time spectrum
        FFT_V_sym = np.zeros([nseg,seglen,nlon],dtype=complex)
        for iseg in range(0,nseg):
            FFT_V_sym[iseg,:,:] = np.fft.fft2(V_sym_seg[iseg,:,:])/(nlon*seglen)*4
                
        # Power spectrum
        C_sym = FFT_V_sym*np.conj(FFT_V_sym) # Power spectrum of PCQ1
        Cm_sym = np.nanmean(C_sym,0)
            
    Cm_sym_shift = np.fft.fftshift( np.fft.fftshift(Cm_sym,axes=1),axes=0 )
    del C_sym, Cm_sym
    
    # apply 1-2-1 filter in frequency
    for i in range(1,seglen-1):
        Cm_sym_shift[i,:] = 1/4*Cm_sym_shift[i-1,:]+1/2*Cm_sym_shift[i,:]+1/4*Cm_sym_shift[i+1,:] 
    
    # Calculate background spectrum and signal strength if KW-proj = 0
    if kw_meridional_proj == 0:
        
        Cm_asy_shift = np.fft.fftshift( np.fft.fftshift(Cm_asy,axes=1),axes=0 )
        del C_asy, Cm_asy
        
        # apply 1-2-1 filter in frequency
        for i in range(1,seglen-1):
            Cm_asy_shift[i,:] = 1/4*Cm_asy_shift[i-1,:]+1/2*Cm_asy_shift[i,:]+1/4*Cm_asy_shift[i+1,:] 
        
        #############################
        # Find background spectrum: 
        #############################
        # 1-2-1 filter in frequency for many times
        fcyc = 15 #How many times applying filters
        fcycs = str(fcyc)
        Cm_shift = (Cm_sym_shift+Cm_asy_shift)/2
        for k in range(0,fcyc):   
            for i in range(1,seglen-1):
                Cm_shift[i,:] = 1/4*Cm_shift[i-1,:] + 1/2*Cm_shift[i,:] + 1/4*Cm_shift[i+1,:] 
        #
        # 1-2-1 filter in wavenum 
        for k in range(0,fcyc):
            for i in range(1,nlon-1):
                Cm_shift[:,i] = 1/4*Cm_shift[:,i-1] + 1/2*Cm_shift[:,i] + 1/4*Cm_shift[:,i+1]   

        #####################################
        # Calculate signal strength = raw/smooth
        r_sym = Cm_sym_shift/Cm_shift
        r_asy = Cm_asy_shift/Cm_shift
    
        
    #Fs_t = 8
    freq = np.arange(-seglen/2,seglen/2)*Fs_t/seglen
    #Fs_lon = 1/2.5
    zonalwnum = np.arange(-nlon/2,nlon/2)*Fs_lon/nlon*360
    x,y = np.meshgrid(zonalwnum, -freq)
    freq = freq*(-1)
    
    power_spectrum = Cm_sym_shift
    
    if kw_meridional_proj == 1:
        return power_spectrum, x, y, freq, zonalwnum, dof
    elif output_sym_only == 1:
        return power_spectrum, r_sym, x, y, freq, zonalwnum, dof
    else:
        # Cm_sym_shift, Cm_asy_shift: symmetric/antisymmetric power of V
        # Cm_shift: background power of V
        # r_sym, r_asy: signal strength of symmetric/antisymmetric of V
        return Cm_sym_shift, Cm_asy_shift, Cm_shift, r_sym, r_asy, x, y, freq, zonalwnum, dof

###########################################################
def plot_raw_sym_spectrum(power_V, x, y, clev, cticks, fig_dir, vname, icase, CASE_SHORT): 
    # vname = 'precip'
    # icase = 0:-4K, 1:CTL, 2:+4K
    # CASESHORT = 'CTL_20y'
    # plot_rym = 0 or 1, whether plotting signal strength or not
    
    #####################
    # Constant:
    g = 9.8
    re = 6371*1000 #earth radius (m)
    s2d = 86400
    
    #######################
    # Set up KW, ER, MJO band
    # 1. KW 
    d = np.array([3,6,20]) # mark 3, 6, 20 day in WK1999
    if icase == 0:
        he = np.array([8,25,50]) 
        hname = list(['8m','25m','50m'])        
    elif icase==1:
        he = np.array([12,25,90]) 
        hname = list(['12m','25m','90m'])
    elif icase == 2:
        he = np.array([25,90,150]) 
        hname = list(['25m','90m','150m'])

    dname = list(['3d','6d','20d'])
    # dispersion curve
    xloc = np.array([12,12,4.9])
    yloc = np.array([0.29,0.47,0.47])
    print('he:',he)
    cp = (g*he)**0.5
    print('Cp:',cp)
    zwnum_goal = 0.5/s2d/cp*2*np.pi*re
    # CCKW band
    s_min = (g*he[0])**0.5/(2*np.pi*re)*s2d #slope of he = 8m
    s_max = (g*he[2])**0.5/(2*np.pi*re)*s2d #slope of he = 90m
    kw_tmax_list = np.array([8,7,4])
    kw_tmax = kw_tmax_list[icase]
    fmax = np.array([0.4,1/2.25,0.5])
    kw_x = np.array([1/kw_tmax/s_max, 1/kw_tmax/s_min,     15,  15, fmax[icase]/s_max, 1/kw_tmax/s_max])
    kw_y = np.array([1/kw_tmax,             1/kw_tmax,  15*s_min, fmax[icase],      fmax[icase],  1/kw_tmax])
    
    '''
    #####################
    # 2. ER -->This will work in .py, uncomment this part if using .py
    # dispersion curve
    h_er = np.array([50,25,1])
    for ih in range(0,np.size(h_er)):
        swfreq,swwn = tool.genDispersionCurves(nWaveType=6, nPlanetaryWave=50, rlat=0,Ahe=[h_er[ih]])
        swfreq = swfreq.squeeze()
        swwn = swwn.squeeze()
        if ih == 0:
            swf = np.empty([6,3,np.size(swfreq,1)])
            swk = np.empty([6,3,np.size(swfreq,1)])              
        swf[:,ih,:] = np.where(swfreq == 1e20, np.nan, swfreq) #(wavetype: 3 for ER, eqdepth, zonalwnum)
        swk[:,ih,:] = np.where(swwn == 1e20, np.nan, swwn)
    ii = 3
    c = 'k'
    # band
    swk_new = np.empty([3,np.size(swk,2)])
    swf_new = np.empty([3,np.size(swk,2)])
    swk_new[:] = np.nan
    swf_new[:] = np.nan
    for iy in range(0,np.size(swk,2)):
        if swk[ii,0,iy]<=-1 and swk[ii,0,iy]>=-10:
            swk_new[0,iy] = swk[ii,0,iy]
            swf_new[0,iy] = swf[ii,0,iy]
        if swk[ii,2,iy]>=-10 and swf[ii,2,iy]>=1/90:
            swk_new[2,iy] = swk[ii,2,iy]
            swf_new[2,iy] = swf[ii,2,iy]
    '''
    ########################
    # 3. MJO band
    mjo_x = np.array([1,      5,      5,  1,  1 ])
    mjo_y = np.array([1/90, 1/90,  1/30, 1/30, 1/90])
    ##############################################################      
    
    ############################
    # Plotting raw spectrum
    #############################
    fig = plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 18})
    plt.contourf(x,y,np.log10(power_V),cmap=get_cmap('hot_r'),levels = clev,extend='both')
    #plt.contourf(x,y,np.log10(power_V),cmap=get_cmap('hot_r'),extend='both')
    cb = plt.colorbar(orientation = 'vertical',shrink=.9)
    cb.set_ticks(cticks)
    # Mark 3, 6, 20 day period:
    for dd in range(0,np.size(d)):
        plt.plot([-15,15], [1/d[dd],1/d[dd]], 'k',linewidth=1, linestyle=':')
        plt.text(-14.8,1/d[dd]+0.01,dname[dd], fontsize=15)
    for hh in range(0,np.size(he)):
        plt.plot([0,zwnum_goal[hh]],[0,0.5],'b',linewidth=1,linestyle=(0,(5,5)))             
    # Mark zwnum == 0:
    plt.plot([0,0],[0,0.5],'k',linewidth=1,linestyle=':')#'dashed')
    # Mark CCKW band:
    for kk in range(0,np.size(kw_x)):
        plt.plot(kw_x,kw_y,'b',linewidth=1.5,linestyle='solid')
    '''
    ###############################
    # Make sure this block works
    # Mark ER dispersion curve
    plt.plot(swk[ii, 0,:], swf[ii,0,:], color=c,linewidth=1,linestyle=(0,(5,5)))
    plt.plot(swk[ii, 1,:], swf[ii,1,:], color=c,linewidth=1,linestyle=(0,(5,5)))
    plt.plot(swk[ii, 2,:], swf[ii,2,:], color=c,linewidth=1,linestyle=(0,(5,5)))
    # Mark ER box
    plt.plot(swk_new[0,:], swf_new[0,:], color='g',linewidth=1.5)
    plt.plot(swk_new[2,:], swf_new[2,:], color='g',linewidth=1.5)
    plt.plot([-10,-10],[0.103,0.026],'g',linewidth=1.5)
    plt.plot([-5.3,-0.5],[1/90,1/90],'g',linewidth=1.5)
    ########################################
    '''
    # Mark MJO box
    for imjo in range(0,np.size(mjo_x)):
        plt.plot(mjo_x,mjo_y,'brown',linewidth=1.5,linestyle='solid')
    #
    plt.title('Raw power (log)-sym: '+vname+', '+CASE_SHORT)
    plt.ylabel('frequency')
    plt.xlabel('zonal wavenumber')
    plt.axis([-15,15,0,0.5])
    plt.xticks([-15,-10,-5,0,5,10,15])
    plt.yticks(np.arange(0,0.55,0.05))
    plt.savefig(fig_dir+vname+'_raw_sym.png')
    plt.show()
    plt.close()                   

###################################################################
def plot_signal_strength_sym(r_sym, x, y, clev, cticks, fig_dir, vname, icase, CASE_SHORT): 
    
    #####################
    # Constant:
    g = 9.8
    re = 6371*1000 #earth radius (m)
    s2d = 86400
    
    #######################
    # Set up KW, MJO band
    # 1. KW 
    d = np.array([3,6,20]) # mark 3, 6, 20 day in WK1999
    if icase == 0:
        he = np.array([8,25,50]) 
        hname = list(['8m','25m','50m'])        
    elif icase==1:
        he = np.array([12,25,90]) 
        hname = list(['12m','25m','90m'])
    elif icase == 2:
        he = np.array([25,90,150]) 
        hname = list(['25m','90m','150m'])

    dname = list(['3d','6d','20d'])
    # dispersion curve
    xloc = np.array([12,12,4.9])
    yloc = np.array([0.29,0.47,0.47])
    cp = (g*he)**0.5
    zwnum_goal = 0.5/s2d/cp*2*np.pi*re
    # CCKW band
    s_min = (g*he[0])**0.5/(2*np.pi*re)*s2d #slope of he = 8m
    s_max = (g*he[2])**0.5/(2*np.pi*re)*s2d #slope of he = 90m
    kw_tmax_list = np.array([8,7,4])
    kw_tmax = kw_tmax_list[icase]
    fmax = np.array([0.4,1/2.25,0.5])
    kw_x = np.array([1/kw_tmax/s_max, 1/kw_tmax/s_min,     15,  15, fmax[icase]/s_max, 1/kw_tmax/s_max])
    kw_y = np.array([1/kw_tmax,             1/kw_tmax,  15*s_min, fmax[icase],      fmax[icase],  1/kw_tmax])
    
    ########################
    # 3. MJO band
    mjo_x = np.array([1,      5,      5,  1,  1 ])
    mjo_y = np.array([1/90, 1/90,  1/30, 1/30, 1/90])
    ##############################################################  
    
    #################################
    # Plotting signal strength
    ###########################
    fig = plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 18})
    plt.contourf(x,y,r_sym,cmap=cmaps.WhiteBlueGreenYellowRed,levels = clev,extend='both')
    cb = plt.colorbar(orientation = 'vertical',shrink=.9)
    cb.set_ticks(cticks)
    
    # Mark 3, 6, 20 day period:
    for dd in range(0,np.size(d)):
        plt.plot([-15,15], [1/d[dd],1/d[dd]], 'k',linewidth=1, linestyle=':')
        plt.text(-14.8,1/d[dd]+0.01,dname[dd], fontsize=15)
    for hh in range(0,np.size(he)):
        plt.plot([0,zwnum_goal[hh]],[0,0.5],'b',linewidth=1,linestyle=(0,(5,5)))             
    # Mark zwnum == 0:
    plt.plot([0,0],[0,0.5],'k',linewidth=1,linestyle=':')#'dashed')
    # Mark CCKW band:
    for kk in range(0,np.size(kw_x)):
        plt.plot(kw_x,kw_y,'b',linewidth=1.5,linestyle='solid')

    # Mark MJO box
    for imjo in range(0,np.size(mjo_x)):
        plt.plot(mjo_x,mjo_y,'brown',linewidth=1.5,linestyle='solid')
    #
    plt.title('Signal stregnth-sym: '+vname+', '+CASE_SHORT)
    plt.ylabel('frequency')
    plt.xlabel('zonal wavenumber')
    plt.axis([-15,15,0,0.5])
    plt.xticks([-15,-10,-5,0,5,10,15])
    plt.yticks(np.arange(0,0.55,0.05))
    plt.savefig(fig_dir+vname+'_rsym.png')
    plt.show()
    plt.close() 


#####################################################################
def calculate_coh2_spectrum(V1, V2, kw_meridional_proj=1, Fs_t=8):
    
    if kw_meridional_proj == 1:
        V1_sym = V1 #(time, lon)
        V2_sym = V2
        nt = np.size(V1,0)
        nlon = np.size(V1,1)
        
    # subset into segments in time (96 days, overlap 60 days)
    seglen = int(96*Fs_t)  
    overlap = int(60*Fs_t)  
    Hw = int(5*Fs_t)   #width of Hann window
    n = int(seglen-overlap)#average seglen (not counting the overlap part)
    nseg = math.floor((nt-seglen)/n)+1
    dof = 2*nseg
    #print('dof:',2*nseg)
    V1_sym_seg = np.zeros([nseg,seglen,nlon])
    V2_sym_seg = np.zeros([nseg,seglen,nlon])

    HANN = np.concatenate((np.hanning(Hw),np.ones(seglen-Hw*2),np.hanning(Hw)),axis=0)
    HANN = np.tile(HANN,(nlon,1))
    HANN = HANN.transpose(1, 0)
    for iseg in range(0,nseg):
        iseg_n = int(iseg*n)
        V1_sym_seg[iseg,:,:] = signal.detrend(V1_sym[iseg*n:iseg*n+seglen,:],axis=0)*HANN
        V2_sym_seg[iseg,:,:] = signal.detrend(V2_sym[iseg*n:iseg*n+seglen,:],axis=0)*HANN

        # calculate space-time spectrum
        FFT_V1_sym = np.zeros([nseg,seglen,nlon],dtype=complex)
        FFT_V2_sym = np.zeros([nseg,seglen,nlon],dtype=complex)
        for iseg in range(0,nseg):
            FFT_V1_sym[iseg,:,:] = np.fft.fft2(V1_sym_seg[iseg,:,:])/(nlon*seglen)*4
            FFT_V2_sym[iseg,:,:] = np.fft.fft2(V2_sym_seg[iseg,:,:])/(nlon*seglen)*4

    A_sym = FFT_V2_sym*np.conj(FFT_V1_sym) # Cross spectrum between V1 and V2
    B_sym = FFT_V1_sym*np.conj(FFT_V1_sym) # Power spectrum of V1
    C_sym = FFT_V2_sym*np.conj(FFT_V2_sym) # Power spectrum of V2

    # average between different segment
    Am_sym = np.nanmean(A_sym,0)
    Bm_sym = np.nanmean(B_sym,0)
    Cm_sym = np.nanmean(C_sym,0)

    Am_sym_shift = np.fft.fftshift( np.fft.fftshift(Am_sym,axes=1),axes=0 )
    Bm_sym_shift = np.fft.fftshift( np.fft.fftshift(Bm_sym,axes=1),axes=0 )
    Cm_sym_shift = np.fft.fftshift( np.fft.fftshift(Cm_sym,axes=1),axes=0 )

    # apply 1-2-1 filter in frequency
    for i in range(1,seglen-1):
        Am_sym_shift[i,:] = 1/4*Am_sym_shift[i-1,:]+1/2*Am_sym_shift[i,:]+1/4*Am_sym_shift[i+1,:] 
        Bm_sym_shift[i,:] = 1/4*Bm_sym_shift[i-1,:]+1/2*Bm_sym_shift[i,:]+1/4*Bm_sym_shift[i+1,:] 
        Cm_sym_shift[i,:] = 1/4*Cm_sym_shift[i-1,:]+1/2*Cm_sym_shift[i,:]+1/4*Cm_sym_shift[i+1,:] 

    coh_sym = np.zeros([seglen,nlon],dtype=float)
    sinph_sym = np.zeros([seglen,nlon],dtype=float)
    cosph_sym = np.zeros([seglen,nlon],dtype=float)

    coh_crit = find_critical_coh2(dof) # Change this into find_critical_coh2(dof) when pasting this whole thing in .py file
        
    coh_sym = np.abs(Am_sym_shift)**2/(Bm_sym_shift*Cm_sym_shift)
    sinph_sym = np.imag(Am_sym_shift)/np.abs(Am_sym_shift)
    cosph_sym = np.real(Am_sym_shift)/np.abs(Am_sym_shift)
        
    coh_sym_new   = np.where(coh_sym>=coh_crit,coh_sym,np.nan)
    sinph_sym_new = np.where(coh_sym>=coh_crit,sinph_sym,np.nan) # we only need the phase information where coh^2>1
    cosph_sym_new = np.where(coh_sym>=coh_crit,cosph_sym,np.nan)
        
    freq = np.arange(-seglen/2,seglen/2)*Fs_t/seglen
    Fs_lon = 1/2.5
    zonalwnum = np.arange(-nlon/2,nlon/2)*Fs_lon/nlon*360
    x,y = np.meshgrid(zonalwnum, -freq)
    freq = freq*(-1)
        
    return coh_sym_new, sinph_sym_new, cosph_sym_new, coh_crit, x, y, freq, zonalwnum


######################################################################################
def KW_wavenumber_frequency_filter(anomalies, zwnum_min, zwnum_max, freq_min, freq_max, equiv_depth_min, equiv_depth_max, Fs_t=8, Fs_lon=1/2.5): # anomalies (time,lon) are KW-meridional-projected. Typical KW band consists of zonal wavenumber 1-14, frequency 1/20-1/2.5 (1/day), equivalent depth is 8-90 (m), Fs_t = data/day, if 3-hourly, Fs_t = 8; if daily, Fs_t = 1.  
    
    # Constant
    g = 9.8       # m/s^2
    
    # Basic information of anomalies: 2D variable, KW-meridional-projection (time, lon)
    nt = np.size(anomalies,0)
    nlon = np.size(anomalies,1)
        
    # make sure nan becomes zero
    if np.sum(np.isnan(anomalies))!=0:
        print('has nan, replace by zero !')
        anomalies_nonan = np.where(np.isnan(anomalies)==1,0,anomalies)
    else:
        anomalies_nonan = anomalies

    # FFT
    V_detrend = signal.detrend(anomalies_nonan,axis=0)

    #############################
    # This is what I add today (2022.08.08)
    # Apply Hann window at the end of timeseries to make sure it is periodic
    Hw = int(0.05*nt) #12
    if Hw % 2==1: #make sure Hw has a width even number
        #print('modify an odd number into an even number')
        Hw = Hw-1
    #print(Hw)
    hann = np.hanning(Hw)
    half = int(Hw/2)
    #print(half)
    HANN = np.concatenate((hann[:half],np.ones(nt-Hw),hann[half:]),axis=0)
    HANN = np.tile(HANN,(nlon,1))
    HANN = HANN.transpose(1,0)
    V_detrend = V_detrend*HANN

    #######################
    # Need to double check this part !!! 2022.8.15
    FFT_V = np.zeros([nt,nlon],dtype=complex)
    FFT_V[:,:] = np.fft.fft2(V_detrend[:,:])  

    V_shift = np.fft.fftshift( np.fft.fftshift(FFT_V,axes=1),axes=0 )   
    V_shift2 = np.zeros([nt,nlon],dtype=complex)
                
    # Frequency filter
    freq = np.arange(-nt/2,nt/2)*Fs_t/nt
    freq_1 = freq_min #1/20 
    freq_2 = freq_max #1/2.5 #10 day
    ifreq_1 = np.abs(freq-freq_1).argmin() 
    ifreq_2 = np.abs(freq-freq_2).argmin()
    ifreq_1_neg = np.abs(freq-(-freq_1)).argmin() 
    ifreq_2_neg = np.abs(freq-(-freq_2)).argmin() 

    # Zonal wavenumber filter
    zwnum = np.arange(-nlon/2,nlon/2)*Fs_lon/nlon*360 #zonal wavenum
    wnum_1 = zwnum_min #1
    wnum_2 = zwnum_max #15
    iwnum_1 = np.abs(zwnum-wnum_1).argmin() 
    iwnum_2 = np.abs(zwnum-wnum_2).argmin() 
    iwnum_1_neg = np.abs(zwnum+wnum_1).argmin() #negative here means positive in reality
    iwnum_2_neg = np.abs(zwnum+wnum_2).argmin()                                     

    # Equivalent depth filter
    hmin = equiv_depth_min #m
    hmax = equiv_depth_max #m
    for ifreq in range(ifreq_1,ifreq_2+1):
        for iwnum in range(iwnum_2_neg,iwnum_1_neg+1):
            f = freq[ifreq]/86400
            k = zwnum[iwnum]/(2*6371*1000*np.pi)
            C = abs(f/k) #phase speed c = f(1/s)/k(1/m) 
            he = C**2/g
            if he<=hmax and he>=hmin:
                V_shift2[ifreq, iwnum]  = V_shift[ifreq,iwnum]  

    for ifreq in range(ifreq_2_neg,ifreq_1_neg+1):
        for iwnum in range(iwnum_1,iwnum_2+1):
            f = freq[ifreq]/86400
            k = zwnum[iwnum]/(2*6371*1000*np.pi)
            C = abs(f/k) #phase speed c = f(1/s)/k(1/m) 
            he = C**2/g
            if he<=hmax and he>=hmin:
                V_shift2[ifreq, iwnum]  = V_shift[ifreq,iwnum] 

    V_shift2 = np.fft.ifftshift( np.fft.ifftshift(V_shift2,axes=1),axes=0 )
    V_filtered = np.real(np.fft.ifft2(V_shift2))

    ######
    # Remove the beginning and the end of timeseries to avoid being affected by tapering
    V_filtered = V_filtered[half:-half,:]    
    time = np.arange(0,nt)
    time_filtered = time[half:-half]

    return V_filtered, time_filtered

######################################################################################
def find_KW_average_freq_zwnum_speed(x, y, r_sym, wnum_min, wnum_max, freq_min, freq_max, h_min, h_max, dof):
    
    # Define constant
    s2d = 86400
    g = 9.8
    earth_perimeter = 2*np.pi*6371*10**3
    
    # Determine critical F value (minimum signal strength r): at significance level =0.01
    rmin = stat.f.ppf(q=1-0.01, dfn=dof, dfd=1000) #dfn: degree of freedom of numerator (raw spectrum), dfd: degree of freedom of denominator (background spectrum, dof=super large)
    print('Critical signal strength value for significance level 0.01: ',rmin) 
    
    k = x/earth_perimeter
    Cp = y/s2d/k
    Cp_min = (g*h_min)**0.5
    Cp_max = (g*h_max)**0.5
    
    r_sym_KW     = np.where(((x>=wnum_min)&(x<=wnum_max)&(y>=freq_min)&(y<=freq_max)&(Cp>=Cp_min)&(Cp<=Cp_max)),r_sym,np.nan)
    r_sym_masked = np.ma.masked_array(r_sym_KW, (np.isnan(r_sym_KW))|(r_sym_KW<rmin)) # not calculating the nan    
    zwnum_ave = np.ma.average(x, weights=r_sym_masked)
    freq_ave  = np.ma.average(y, weights=r_sym_masked)
    He_ave    = ( freq_ave/s2d/zwnum_ave*earth_perimeter )**2/g
    Cp_ave    = ( g*He_ave )**0.5
    
    zwnum_ave = np.real(zwnum_ave)
    freq_ave  = np.real(freq_ave)
    He_ave    = np.real(He_ave)
    Cp_ave    = np.real(Cp_ave)
    
    return zwnum_ave, freq_ave, He_ave, Cp_ave


############################################
def KW_meridional_projection(V, lat, tropics_or_midlat, lat_0=9, lat_tropics=10):
    # Variable is (time, plev, lat, lon) or (time, lat, lon) or (time, lat) or (lat), tropics_or_midlat =0:tropics only, =1:midlat only
    # Ignore nan data when doing projection, the projected outcome should be without nan

    KW_coef = np.exp( -(lat/(2*lat_0))**2)
    if tropics_or_midlat == 0:
        tropics_midlat_filter = np.where(np.abs(lat)>lat_tropics,0,1)
    else:
        tropics_midlat_filter = np.where(np.abs(lat)>lat_tropics,1,0)

    Vdim = np.size(np.shape(V))
    KW_filt = KW_coef*tropics_midlat_filter
    SUM = np.sum(KW_filt)

    if Vdim ==4: # Variable is (time, plev, lat, lon)
        nt   = np.size(V,0)
        nlev = np.size(V,1)
        nlon = np.size(V,3)
        V_T = np.transpose(V, (0,1,3,2)) #(time, plev, lon, lat)
        del V
    elif Vdim ==3: # Variable is (time, lat, lon)
        nt   = np.size(V,0)
        nlon = np.size(V,2)
        V_T = np.transpose(V, (0,2,1))
        del V
    elif Vdim ==2: # Variable is (time, lat)
        nt  = np.size(V,0)
        V_T = V
        del V
    elif Vdim ==1:
        V_T = V
        del V

    if np.sum(np.isnan(V_T))==0:
        V_projected = np.inner(V_T, KW_filt)/SUM
        del V_T
    else:
        if Vdim == 3 or Vdim == 4:
            KW_filt_large = np.tile(KW_filt, (nt, nlon, 1))
        elif Vdim == 2:
            KW_filt_large = np.tile(KW_filt, (nt, 1))
        elif Vdim == 1:
            KW_filt_large = KW_filt

        if Vdim != 4:
            KW_filt_mask = np.ma.array(KW_filt_large, mask=np.isnan(V_T))
            del KW_filt_large
            Vsum = np.nansum(V_T*KW_filt_mask, Vdim-1)
            del V_T
            KWsum = np.nansum(KW_filt_mask, Vdim-1)
            del KW_filt_mask
            V_projected  = Vsum/KWsum
            del Vsum, KWsum
        else: # 4D data is to large, and therefore I calculate the meridional projection by each level. By this way, we can separate the vertical levels with and without nan. Where there are no nans in some levels, direct inner product will be faster.
            V_projected = np.empty([nt,nlev,nlon])
            for ilev in range(0,nlev):
                print('Lev: ',ilev)
                if np.sum(np.isnan(V_T[:,ilev,:,:]))==0:
                    V_projected[:,ilev,:] = np.inner(V_T[:,ilev,:,:], KW_filt)/SUM
                else:
                    KW_filt_mask = np.ma.array(KW_filt_large, mask = np.isnan(V_T[:,ilev,:,:]))
                    Vsum = np.nansum(V_T[:,ilev,:,:]*KW_filt_mask, Vdim-2)
                    KWsum = np.nansum(KW_filt_mask, Vdim-2)
                    del KW_filt_mask
                    V_projected[:,ilev,:] = Vsum/KWsum
                    del Vsum, KWsum

    return V_projected



######################################
def calculate_KW_phase(V, vname, unit, figdir, Nstd = 1, plot_fig = 1, dph = 1/8*np.pi):
    # Output: V_kw_phase_composite (phase), phase_bin, phase_correct
    
    # Basic info
    nt    = np.size(V,0)
    nlon  = np.size(V,1)
    V_std = np.nanstd(V)
    #print('Std of variable:',V_std)
    
    # Parameters used here
    pi  = np.pi
    mybin = np.arange(-pi, pi+dph*2, dph)-dph/2 #play around this 1/8
    phase_bin = mybin[:-1] + dph/2
    std = str(Nstd)+'std'
    PI = '\u03C0'
    bin_simple = np.arange(-pi,pi+1/4*pi,1/4*pi)

    ################
    # Step2: Find local maximum and minimum and the nearest peak for each instant
    ##################
    #  Only consider phase if the nearest local maximum or minimum is greater than V_std*Nstd
    # Find index of the local maxmium and minimum
    local_min_max_id = np.empty([nt,nlon])
    local_min_max_id[:] = np.nan
    local_min_max_id[1:-1,:] = np.where( (V[1:-1,:]<=V[2:,:]) & (V[1:-1,:]<=V[0:-2,:]), 1, np.nan ) #local min
    local_min_max_id[1:-1,:] = np.where( (V[1:-1,:]>=V[2:,:]) &(V[1:-1,:]>=V[0:-2,:]), -1, local_min_max_id[1:-1,:] ) #local max

    #***********************
    # For each instant, assign the nearest peak of the same sign
    V_peak = np.empty([nt,nlon])
    V_peak[:] = np.nan

    const = np.array([1,-1])
    for ilon in range(0,nlon):

        zero_idx = np.argwhere(V[:-1,ilon]*V[1:,ilon]<=0).squeeze()
        peak_idx = np.argwhere(np.abs(local_min_max_id[:,  ilon])==1).squeeze() #local min/max

        # for every local min/max, find the nearest zero or local min
        nmax = np.size(peak_idx)
        for i in range(0,nmax): 
            # Only select strong peak, do not consider weak peak
            if (np.abs(V[peak_idx[i],ilon])<V_std*Nstd): 
                continue
            dpeak_zero = zero_idx-peak_idx[i]
            dpeak_zero_p = np.where(dpeak_zero>=0, dpeak_zero, np.nan) #positive
            dpeak_zero_n = np.where(dpeak_zero<0, dpeak_zero, np.nan) #negative

            # Find out the right boundary:
            tmp = np.argwhere( (dpeak_zero>=0) & (dpeak_zero==np.nanmin(dpeak_zero_p)) ).squeeze()
            if np.size(tmp)==0: #this means tmp is empty
                id_r = peak_idx[i]
            else:
                id_tmp = zero_idx[tmp]
                if i!=nmax-1:
                    if id_tmp>peak_idx[i+1]:
                        id_r = peak_idx[i+1]-1
                    else:
                        id_r = id_tmp
                else:
                    id_r = id_tmp

            # Find out the left boundary
            tmp = np.argwhere( (dpeak_zero<0) & (dpeak_zero==np.nanmax(dpeak_zero_n)) ).squeeze()
            if np.size(tmp)==0: #this means temp is empty
                id_l = peak_idx[i]
            else:
                id_tmp = zero_idx[tmp]+1
                if i!=0:
                    if id_tmp<peak_idx[i-1]:
                        id_l = peak_idx[i-1]+1
                    else:
                        id_l = id_tmp
                else:
                    id_l = id_tmp

            V_peak[ int(id_l):int(id_r+1),  ilon] = V[peak_idx[i], ilon]

    # Find whether this is the enhanced or decaying phase (enhanced phase:0, decaying phase:1)
    enh_dec = np.empty([nt,nlon])
    enh_dec[:] = np.nan
    # Enhanced phase: V value is "larger" than the previous day and "smaller" than the next day
    enh_dec[1:-1,:] = np.where( ((V[1:-1,:]>V[0:-2,:]) & (V[1:-1,:]<V[2:,:]) & (np.isnan(V_peak[1:-1,:])==0)), 0, np.nan  ) 
    # Decaying phase: V value is "smaller" than the previous day and "larger" than the next day
    enh_dec[1:-1,:] = np.where( ((V[1:-1,:]<V[0:-2,:]) & (V[1:-1,:]>V[2:,:]) & (np.isnan(V_peak[1:-1,:])==0)), 1, enh_dec[1:-1,:]  ) 

    # For each instant, determine whether it is the enhanced and decaying phase
    enh = V_peak*(1-enh_dec)
    dec = V_peak*enh_dec
    enh = np.where(enh==0,np.nan,enh)
    dec = np.where(dec==0,np.nan,dec)

    ################
    # Test Step2: Check V timeseries and V_peak
    #################
    t = np.arange(0,200)#2100,2300)
    zero = np.zeros([np.size(t)])
    if plot_fig == 1:
        fig_name = 'KW_'+vname+'_time_evolution_sample_enh_dec'+std+'.png'
        fig = plt.subplots(1,1,figsize=(3.2, 2.4),dpi=600)
        plt.subplots_adjust(left=0.2,right=0.95,top=0.95,bottom=0.16)
        plt.rcParams.update({'font.size': 7})
        plt.plot(t, V[t,10],'b-o',markersize=4)
        plt.plot(t, V_peak[t,10],'g-o',markersize=4)
        plt.plot(t, enh[t,10],'m-o',markersize=4)
        plt.plot(t, dec[t,10],'y-o',markersize=4)
        plt.legend([vname+' kw','strong peak','growing','decaying'])
        plt.plot(t, zero, 'k--')
        plt.xlabel('time (days since yr03)')
        plt.ylabel('KW-filtered '+vname)
        plt.savefig(figdir+fig_name,format='png', dpi=600)
        #plt.show()
        plt.close()

    ############################
    # Step 3: Calculate wave phase
    ###########################
    # Normalize the variable value with the closest peak V values for the active and inactive phase.
    V_norm = V/np.abs(V_peak)
    # Calculate the phase
    phase = np.arcsin(V_norm)
    phase = np.where( np.isnan(V_peak)==0, phase, np.nan)
    phase_corr  = np.where( ((enh_dec==1) & (V_peak<=0)), -np.pi-phase, phase) # dec + prpeak <0: (-pi~-pi/2), new_theta = -pi-theta
    phase_correct = np.where( ((enh_dec==1) & (V_peak>=0)), np.pi-phase,  phase_corr) # dec + prpeak >0: (pi/2~pi), new_theta = pi-theta
    phase_correct = -phase_correct
    #print('=============================================')
    #print('Should be pi and -pi')
    #print(np.nanmax(phase_correct), np.nanmin(phase_correct)) 

    ######################
    # Test step3: Check wave_phase assign is correct
    ######################
    kw_active   = np.where(phase_correct==-np.pi/2, V, np.nan)
    kw_inactive = np.where(phase_correct==np.pi/2, V, np.nan)

    t = np.arange(0,200) #t = np.arange(2100,2300)
    zero = np.zeros([np.size(t)])
    half_p = 1/2*np.pi*np.ones([np.size(t)])
    half_n = -1/2*np.pi*np.ones([np.size(t)])
    #
    if plot_fig == 1:
        fig_name = 'KW_V_time_evolution_sample_'+std+'.png'
        fig = plt.subplots(1,1,figsize=(3.2, 3.2),dpi=600)
        plt.subplots_adjust(left=0.2,right=0.99,top=0.95,bottom=0.13,hspace=0.24)
        plt.rcParams.update({'font.size': 7})
        plt.subplot(2,1,1)
        plt.plot(t,V[t,10],'b-o',markersize=2)
        plt.plot(t,kw_active[t,10],'mo',markersize=2)
        plt.plot(t,kw_inactive[t,10],'go',markersize=2)
        plt.plot(t,zero,'k:')
        plt.ylabel('KW-filtered '+vname)
        plt.legend(['KW '+vname,'active','inactive'])
        plt.subplot(2,1,2)
        plt.plot(t,phase_correct[t,10],'c-o',markersize=2)
        plt.plot(t,zero,'k:')
        plt.plot(t,half_p,'k:')
        plt.plot(t,half_n,'k:')
        plt.xlabel('time (days since yr03)')
        plt.ylabel('KW-filtered '+vname)
        plt.yticks(np.arange(-1,1.5,0.5)*np.pi,('-'+PI,'-1/2'+PI,'0','1/2'+PI,PI))
        plt.ylim([-np.pi,np.pi])
        plt.savefig(figdir+fig_name,format='png', dpi=600)
        #plt.show()
        plt.close()

    ####################
    # Step4: Composite other variables based on the phase. 
    ##################
    # Make sure there is no nan
    phase_flat = np.ndarray.flatten(phase_correct)
    phase_flat_nonan = np.delete(phase_flat, np.argwhere(np.isnan(phase_flat)==1) )
    ndata_nonan = np.size(phase_flat_nonan)

    # Count how many points for each bin. 
    N_    = stat.binned_statistic(phase_flat_nonan, phase_flat_nonan, statistic='count', bins=mybin)
    N = N_.statistic
    #print(N)

    # Plot how many data for each phase in a KW cycle
    if plot_fig == 1:
        fig_name = 'KW_data_count_'+vname+'_'+std+'.png'
        fig = plt.subplots(1,1,figsize=(3.2, 2),dpi=600)
        plt.subplots_adjust(left=0.2,right=0.99,top=0.9,bottom=0.2)#,hspace=0.24)
        plt.rcParams.update({'font.size': 7})
        plt.plot(phase_bin,N,'m-o',markersize=5)
        plt.xticks(bin_simple,('-'+PI,'-3/4'+PI,'-1/2'+PI,'-1/4'+PI,'0','1/4'+PI,'1/2'+PI,'3/4'+PI,PI))
        plt.legend(['Number of data'])
        plt.ylabel('#')
        plt.xlabel('KW Phase')
        plt.savefig(figdir+fig_name,format='png', dpi=600)
        #plt.show()
        plt.close()

    ###############
    # PLot evolution of V in a KW cycle
    temp = np.ndarray.flatten(V)
    V_nonan = np.delete(temp, np.argwhere(np.isnan(phase_flat)==1) )
    Mean_        = stat.binned_statistic(phase_flat_nonan, V_nonan, statistic='mean', bins=mybin)
    V_kw_phase_composite    = Mean_.statistic
    nbin = np.size(phase_bin)
    zero = np.zeros([nbin])
    #
    if plot_fig == 1:
        fig_name = 'KW_'+vname+'_1cyc.png'
        fig = plt.subplots(1,1,figsize=(3.2, 2),dpi=600)
        plt.subplots_adjust(left=0.2,right=0.99,top=0.95,bottom=0.2)
        plt.plot(phase_bin,V_kw_phase_composite,'b-o',markersize=5)
        plt.plot(phase_bin,zero,'k--')
        plt.rcParams.update({'font.size': 7})
        plt.xticks(bin_simple,('-'+PI,'-3/4'+PI,'-1/2'+PI,'-1/4'+PI,'0','1/4'+PI,'1/2'+PI,'3/4'+PI,PI))
        plt.legend([vname])
        plt.xlabel('KW Phase')
        plt.ylabel('KW-filtered '+vname+', unit:'+unit)
        plt.savefig(figdir+fig_name,format='png', dpi=600)
        #plt.show()
        plt.close()
    
    
    return V_kw_phase_composite, phase_bin, phase_correct


def composite_by_kw_phase(V, kw_phase, time, std=0): #time is for KW-filtered timeseries, which is used to calculate kw_phase, and thus variable needs to have the same time dimension as the KW-phase
    
    # Parameters used
    pi  = np.pi
    dph = 1/8*pi
    mybin      = np.arange(-pi, pi+dph*2, dph)-dph/2 
    bin_center = mybin[:-1] + dph/2
    nbin       = np.size(bin_center)
    
    # Basic info of V
    if np.size(np.shape(V))==2:
        dimsize = 2 # 2-Dimensional (time, lon), ex: precipitable water, precipitation
        nlon = np.size(V,1)
    elif np.size(np.shape(V))==3:
        dimsize = 3 # 3-Dimensional (time, plev, lon), ex:u, v, w, Q, T....
        nlev = np.size(V,1)
        nlon = np.size(V,2)
    nt = np.size(kw_phase,0)
    
    # Make sure Var_p and phase_kw has the same time range
    itmin = int(time[0]) 
    itmax = int(time[-1])
    if np.size(V,0)!=np.size(time):
        if dimsize == 2:
            V_new = V[itmin:itmax+1,:]
        elif dimsize == 3:
            V_new = V[itmin:itmax+1,:,:]
    else:
        V_new = V
    
    # Initialize
    if dimsize == 2:
        V_kw  = np.empty([nbin]) # KW composite (phase)
    elif dimsize == 3:
        V_kw  = np.empty([nbin, nlev]) # KW composite (phase, nlev)
    
    if std == 1:
        if dimsize == 2:
            V_kw_std  = np.empty([nbin]) # KW composite (phase)
        elif dimsize == 3:
            V_kw_std  = np.empty([nbin, nlev]) # KW composite (phase, nlev)



    if np.size(V_new,0)!=nt:
        print('Error: time dimension is different for V and kw_phase!')
     
    # Composite V with respect to the wave phase
    phase_kw_flat = np.reshape(kw_phase,(nt*nlon))
    nonan = np.where( (np.isnan(phase_kw_flat)==0) )
    nonan = np.array(nonan).squeeze()
    phase_nonan = phase_kw_flat[nonan].squeeze()
    
    if dimsize == 2:
        V_new   = np.reshape(V_new,(nt*nlon))
        V_nonan = V_new[nonan] # timeslots with strong KW
        Mean_   = stat.binned_statistic(phase_nonan, V_nonan, statistic='mean', bins=mybin)
        V_kw    = Mean_.statistic
        if std == 1:
            Std_ = stat.binned_statistic(phase_nonan, V_nonan, statistic='std', bins=mybin)
            V_kw_std = Std_.statistic
        
    elif dimsize == 3:
        V_new   = np.transpose(V_new,(0,2,1)) #(nt,nlon,nlev)
        V_new   = np.reshape(V_new,(nt*nlon,nlev))            
        V_nonan = V_new[nonan,:] # timeslots with strong KW
        #
        for ilev in range(0,nlev):
            Mean_        = stat.binned_statistic(phase_nonan, V_nonan[:,ilev], statistic='mean', bins=mybin)
            V_kw[:,ilev] = Mean_.statistic
            if std == 1:
                Std_ = stat.binned_statistic(phase_nonan, V_nonan[:,ilev], statistic='std', bins=mybin)
                V_kw_std[:,ilev] = Std_.statistic

    
    del V_new, V_nonan
    
    if std == 0:
        return V_kw, bin_center
    else:
        return V_kw, V_kw_std, bin_center

###########################################################
def Q_vert_mode_to_WU(EOF1_Q_raw, EOF2_Q_raw, Sp):

    # Goal of this function is to convert EOF modes of Q to vertical modes of omega and u
    # vertical modes of omega are obtained from Q mode divided by static stability
    # vertical modes of u are obtained from vertical derivative of vertical modes of omega
    # raw means no smoothing, Sp: mean static stability profile Sp(plev)
    
    if np.size(Sp)!=np.size(EOF1_Q_raw) or np.size(Sp)!=np.size(EOF2_Q_raw):
        print('Error: Size mismatches between EOFQ and Sp')
    else: 
        EOF1_W = EOF1_Q_raw/Sp # W denotes omega
        EOF2_W = EOF2_Q_raw/Sp

        EOF1_W_smooth = smoothing_121(EOF1_W)
        EOF2_W_smooth = smoothing_121(EOF2_W)

        EOF1_W_smooth = EOF1_W_smooth/(np.sum(EOF1_W_smooth**2))**0.5
        EOF2_W_smooth = EOF2_W_smooth/(np.sum(EOF2_W_smooth**2))**0.5

        EOF1_U_smooth = EOF1_W_smooth[2:]-EOF1_W_smooth[:-2]
        EOF2_U_smooth = EOF2_W_smooth[2:]-EOF2_W_smooth[:-2]

        EOF1_U_smooth = EOF1_U_smooth/(np.sum(EOF1_U_smooth**2))**0.5
        EOF2_U_smooth = EOF2_U_smooth/(np.sum(EOF2_U_smooth**2))**0.5

        return EOF1_W_smooth, EOF2_W_smooth, EOF1_U_smooth, EOF2_U_smooth

        
############################################################################
def vertical_mode_decomposition(V_kw, EOF1, EOF2):
    # V_kw: KW composite vertical structure (phase, nlev), Both V_kw, EOF1/EOF2 only contain tropospheric data
    
    # Make sure vertical dimension of V_kw and EOF1, EOF2 are the same
    if np.size(V_kw,1)!=np.size(EOF1) or np.size(V_kw,1)!=np.size(EOF2):
        print('Error: Vertical dimension is different betweeen V_kw and EOF1/EOF2')
    
    # Basic info
    nphase = np.size(V_kw,0)
    nlev   = np.size(V_kw,1)
    
    # Initialized output variables
    V_kw1    = np.empty([nphase,nlev])
    V_kw2    = np.empty([nphase,nlev])
    
    # Vertical mode decomposition
    V_PC1      = np.dot( V_kw, EOF1 ) 
    V_PC1_lev  = np.tile( V_PC1, (nlev,1)).T
    tmp        = np.tile( EOF1, (nphase, 1) )
    V_kw1[:,:] = V_PC1_lev*tmp
    
    V_PC2      = np.dot( V_kw, EOF2 ) 
    V_PC2_lev  = np.tile( V_PC2, (nlev,1)).T
    tmp        = np.tile( EOF2, (nphase, 1) )
    V_kw2[:,:] = V_PC2_lev*tmp
    
    return V_kw1, V_kw2

#########################################################################
def mca(x,y): #x=x(structure dim M, sampling dim N), y(structure dim L, sampling dim N)
    nt = np.size(x,1)
    C = np.mat(x)*np.mat(y.T)/nt/np.std(x)/np.std(y) #Correlation matrix (M,L)
    
    u, s, v = np.linalg.svd(C, full_matrices=False)
    EOF_x = np.transpose(u) # (M,M), EOFi=EOF[i-1] EOF1=EOF[0],EOF2=EOF[1],....
    PC_x = np.dot(EOF_x,x) # (M,N), PCi =pc[i-1], pc1=pc[0],pc2=pc[1],...
    
    EOF_y = np.transpose(v) # (L,L)
    PC_y = np.dot(EOF_y,y) # (L,N)
    
    eigval = s**2/nt
    eigval_explained_var = eigval/np.sum(eigval)*100 #percent
    
    EOF_x = np.array(EOF_x)
    EOF_y = np.array(EOF_y)
    PC_x  = np.array(PC_x)
    PC_y  = np.array(PC_y)    

    return EOF_x, EOF_y, PC_x, PC_y, eigval, eigval_explained_var

####################################################
def calculate_plot_eof(Va, plev, Q_or_omega, figdir):
    # Q_or_omega = 0-->Q, =1-->Omega
    
    # Define constant
    nt = np.size(Va,0)
    nlev = np.size(Va,1)
    nlon = np.size(Va,2)
    plev_half = 500
    
    if np.sum(np.isnan(Va))!=0:
        print('Error: NAN in Va, cannot calculate EOF')
    
    V_new = np.transpose(Va,(1,0,2))
    V_new = np.reshape(V_new,[nlev,nt*nlon])
        
    EOF, PC, eigval, expvar, err, dof, phi_0, phi_L = MJO.eof(V_new)
    dlev = np.abs(plev-plev_half)
    ilev_half = np.argwhere(dlev==np.min(dlev)).squeeze()   

    if Q_or_omega == 1: #omega
            
        if np.mean(EOF[0,:])> 0.0:
            EOF1 =  -EOF[0,:]
            #print('EOF1 change sign')
            PC1  = -PC[0]
        else:
            EOF1 = EOF[0,:]
            #print('EOF1 not change sign')
            PC1 = PC[0]
        if np.mean(EOF[1,0:ilev_half])>0.0:
            EOF2 = -EOF[1,:]
            #print('EOF2 change sign')
            PC2 = -PC[1]
        else:
            EOF2 = EOF[1,:] 
            #print('EOF2 not change sign')
            PC2 = PC[1]
                
    elif Q_or_omega == 0: #Q
            
        if np.mean(EOF[0,:])> 0.0:
            EOF1 =  EOF[0,:]
            #print('EOF1 not change sign')
            PC1 = PC[0]
        else:
            EOF1 = -EOF[0,:]
            #print('EOF1 change sign')
            PC1 = -PC[0]
        if np.mean(EOF[1,0:ilev_half])>0.0:
            EOF2 = EOF[1,:]
            #print('EOF2 not change sign')
            PC2 = PC[1]
        else:
            EOF2 = -EOF[1,:] 
            #print('EOF2 not change sign')
            PC2 = -PC[1]            
        
    PC1_reshape = np.reshape(PC1,(nt,nlon))
    PC2_reshape = np.reshape(PC2,(nt,nlon))

    expvar1 = "%.1f" %expvar[0]
    expvar2 = "%.1f" %expvar[1]
    zero = np.zeros([np.size(plev)])

    ################################################
    # plot eof
    fig = plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 22})
    plt.plot(EOF1,plev,'g',EOF2,plev,'orange',linewidth=4)
    plt.plot(zero,plev,'k:')
    ax = plt.gca()
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(axis="both", direction="in")
    plt.xlabel('Normalized profiles')
    plt.ylabel('Pressure (hPa)')
    plt.legend(('EOF1 ('+expvar1+'%)','EOF2 ('+expvar2+'%)'),loc='upper left')
    plt.axis([-0.5,0.5,100,1000])
    plt.gca().invert_yaxis()
    ax.set_yticks(np.arange(900,0,-100))
    plt.xticks(np.arange(-0.5,0.6,0.1))
    if Q_or_omega == 0:
        plt.savefig(figdir+'Q_EOF.png')
    else:
        plt.savefig(figdir+'W_EOF.png')
    #plt.show()
    
    return EOF1, EOF2, PC1_reshape, PC2_reshape


##########################################
def rotate_eof(EOF1,EOF2,plev, is_CTL, figdir):

    # Define constants
    dth = 1/32*np.pi
    theta_small = np.arange(-1/4*np.pi,1/4*np.pi+dth, dth)
    nth_small = np.size(theta_small)
    nlev = np.size(plev)
    PI = '\u03C0'
    theta_str = list(['-8'+PI+'/32','-7'+PI+'/32','-6'+PI+'/32','-5'+PI+'/32','-4'+PI+'/32','-3'+PI+'/32',\
                  '-2'+PI+'/32','-'+PI+'/32','0',PI+'/32',PI+'/32','2'+PI+'/32','3'+PI+'/32',\
                 '4'+PI+'/32','5'+PI+'/32','6'+PI+'/32','7'+PI+'/32','8'+PI+'/32'])
 
    for ith in range(0,nth_small):
        th = theta_small[ith]
        
        if ith == 0:
            R_EOF1 = np.empty([nth_small,nlev])
            R_EOF2 = np.empty([nth_small,nlev])

        R_EOF1[ith,:] = np.cos(th)*EOF1-np.sin(th)*EOF2
        R_EOF2[ith,:] = np.sin(th)*EOF1+np.cos(th)*EOF2
        
        len_EOF1 = (np.sum(EOF1**2))**0.5
        len_EOF2 = (np.sum(EOF2**2))**0.5
        len_REOF1 = (np.sum(R_EOF1[ith,:]**2))**0.5
        len_REOF2 = (np.sum(R_EOF2[ith,:]**2))**0.5

        if np.inner(R_EOF1[ith,:],R_EOF2[ith,:])>=10**(-10) or np.abs(len_REOF1-1)>=10**(-5) or np.abs(len_REOF2-1)>=10**(-5):
            print('error!, lenEOF1:',len_REOF1,'lenEOF2:',len_REOF2)

    ########################
    # Use column integrated EOF1 to find the optimal rotation angle
    #############################
    col_EOF1 = MJO.z_integrate_2d(R_EOF1[:,:],plev)
    fig = plt.figure(figsize=(3.2, 2))
    plt.rcParams.update({'font.size': 7})
    plt.plot(theta_small,col_EOF1,'k-o')
    plt.ylabel('Column EOF1')
    plt.xlabel('Rotation angle \u03B8')
    plt.xticks(theta_small[:nth_small+2:2],theta_str[:nth_small+2:2])
    plt.savefig(figdir+'EOF1_col_integrate.png',format='png', dpi=600)
    #plt.show()
    plt.close()
        
    #################################
    # Determine the rotation angle based on maximum column integrated EOF1
    ###################################
    ith_final = np.argwhere(col_EOF1==np.max(col_EOF1)).squeeze()
    
    print('Theta before adjustment:',theta_small[ith_final])
    ####################################
    # Caution: for CTL output, theta = 4pi/32 and 5pi/32 gives similar col_EOF1, however, larger KW variance of the second mode is explained by 5pi/32. We use 5pi/32 instead of 4pi/32 
    ####################################
    if is_CTL == 1 and theta_small[ith_final]!= 5*np.pi/32:
        ith_final = 13 # ith_final=ith_final+1,  theta_small[ith_final=13]=5pi/32
    #######################################

    theta_final  = theta_small[ith_final]
    R_EOF1_final = R_EOF1[ith_final,:]
    R_EOF2_final = R_EOF2[ith_final,:]

    ith0 = np.argwhere(theta_small==0).squeeze()

    print('Theta final:',theta_final)
    #===========================
    # Plot results
    ######################
    # 1. Plot EOF modes
    ##################
    # Unrotated EOF
    zeros = np.zeros([nlev])
    fig = plt.figure(figsize=(3.2, 2.4))
    plt.rcParams.update({'font.size': 7})
    plt.subplot(1,2,1)
    plt.plot(R_EOF1[ith0,:], plev, 'b')
    plt.plot(R_EOF2[ith0,:], plev, 'r')
    plt.xlabel('Original EOF')
    plt.ylabel('plev (hPa)')
    plt.plot(zeros,plev,'k:')
    plt.ylim([np.min(plev),np.max(plev)])
    plt.gca().invert_yaxis()
    # Rotated EOF
    plt.subplot(1,2,2)
    plt.plot(R_EOF1_final[:], plev, 'b')
    plt.plot(R_EOF2_final[:], plev, 'r')
    plt.xlabel('Rotated EOF')
    plt.ylabel('plev (hPa)')
    plt.plot(zeros,plev,'k:')  
    plt.ylim([np.min(plev),np.max(plev)])
    plt.gca().invert_yaxis()
    plt.legend(('EOF1','EOF2'),bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.tight_layout()
    plt.savefig(figdir+'EOF_final.png',format='png', dpi=600)
    #plt.show()
    plt.close()
    
    return R_EOF1_final, R_EOF2_final, th


#################################
# Below is to calculate Cp1, Cp2
#################################
def find_plev_tropopause(T_KW, plev, ilev_STbot, ilev_300): #T_KW(phase, nlev)
    # find lower boundary of TTL by the minimum variance of temperature in KW composite with respect to phase
    # ilev_300 = level of 300 hPa, this is used as we expect tropopause will not be lower than 300hPa
    T_var = np.std(T_KW,0)
    plt.plot(T_var,plev,'b-o')
    plt.gca().invert_yaxis()
    #plt.show()
    ilev_TTL = int(np.argwhere(T_var==np.min(T_var[ilev_STbot:ilev_300])).squeeze())
    plev_TTL = plev[ilev_TTL]
    
    return ilev_TTL, plev_TTL

#####################################
def calculate_PCWS(Tproj, Zproj, Wproj, EOF1_Q, EOF2_Q, plev, remove_10d=1):
    # This code is to calculate PC1 and PC2 for adiabatic cooling and heating Sp*Omega
    # Tproj is (time, plev, lon)
    # remove_10d=1 only for CAM6 model
    # remove_10d=0 for MPAS
    
    # Define constant
    nt   = np.size(Tproj, 0)
    nlev = np.size(Tproj, 1)
    nlon = np.size(Tproj, 2)
    g  = 9.8
    Rd = 287
    Cp = 1004
    gamma_d = g/Cp
    plev_large = np.tile(plev,(nt,nlon,1))
    plev_large = np.transpose(plev_large,(0,2,1))
    
    # Calculate Sp
    gamma = -(Tproj[:,2:,:]-Tproj[:,:-2,:])/(Zproj[:,2:,:]-Zproj[:,:-2,:]) #(nlev-2)
    Sp = (gamma_d-gamma)*Rd*Tproj[:,1:-1,:]/(plev_large[:,1:-1,:]*100)/g   
    
    Spw = Sp*Wproj[:,1:-1,:] #(time, plev, lon)
    Spw = np.transpose(Spw, (0,2,1))
    if remove_10d == 1:
    	Spw = MJO.remove_10d_from_3hr_data(Spw)
    Spw = np.reshape(Spw,(nt*nlon,nlev-2))
    
    PC1_WS = np.matmul(Spw, EOF1_Q[1:-1])
    PC2_WS = np.matmul(Spw, EOF2_Q[1:-1])
    PC1_WS = np.reshape(PC1_WS, (nt, nlon))
    PC2_WS = np.reshape(PC2_WS, (nt, nlon))
    
    return PC1_WS, PC2_WS

#######################################
def calculate_Cp1_Cp2(Tm, zm, ilev_TTL, N2, EOFQ1, EOFQ2, PCQ1_raw, PCQ2_raw, PCWS1_raw, PCWS2_raw, time_kw, phase_kw):
    # Tm, zm, N2 is the mean state temperature, geopotential height, static stability (plev)
    # EOFQ1, EOFQ2 is after rotation
    # PCQ1_KW is PCQ1 selecting only strong KW cases, make sure PCQ and PCWS in the same unit (K/s)
    
    # Define constant
    Rd = 287
    g  = 9.8
    Hs = Rd*np.nanmean(Tm)/g
    
    # Step 1: Calculate Lz
    Lz  = zm[ilev_TTL]*2
    Lz1 = Lz
    Lz2 = Lz/2
    mm1 = 2*np.pi/Lz1
    mm2 = 2*np.pi/Lz2
    
    # Step 2: Weight N^2 by EOF1 and EOF2
    N2_1 = np.average(N2, weights=np.abs(EOFQ1))
    N2_2 = np.average(N2, weights=np.abs(EOFQ2))
    
    # Step 3: Estimate alpha (a)
    # Make sure date of PC is the same as date of phase_kw
    if np.size(PCQ1_raw,0)!=np.size(time_kw):
        itmin = int(time_kw[0])
        itmax = int(time_kw[-1])
        PCQ1_raw  = PCQ1_raw[itmin:itmax+1,:] #(time, lon)
        PCQ2_raw  = PCQ2_raw[itmin:itmax+1,:]
        PCWS1_raw = PCWS1_raw[itmin:itmax+1,:]
        PCWS2_raw = PCWS2_raw[itmin:itmax+1,:]
    
    # Remove non KW days
    PCQ1_KW = np.where(np.isnan(phase_kw)==1, np.nan, PCQ1_raw)
    PCQ2_KW = np.where(np.isnan(phase_kw)==1, np.nan, PCQ2_raw)
    
    PCWS1_KW = np.where(np.isnan(phase_kw)==1, np.nan, PCWS1_raw)
    PCWS2_KW = np.where(np.isnan(phase_kw)==1, np.nan, PCWS2_raw)   
    PCQ1_KW  = np.ndarray.flatten(PCQ1_KW)
    PCQ1_KW  = PCQ1_KW[~np.isnan(PCQ1_KW)]
    PCQ2_KW  = np.ndarray.flatten(PCQ2_KW)
    PCQ2_KW  = PCQ2_KW[~np.isnan(PCQ2_KW)]
    PCWS1_KW = np.ndarray.flatten(PCWS1_KW)
    PCWS1_KW = PCWS1_KW[~np.isnan(PCWS1_KW)]
    PCWS2_KW = np.ndarray.flatten(PCWS2_KW)
    PCWS2_KW = PCWS2_KW[~np.isnan(PCWS2_KW)]
    
    X1 = sm.add_constant(PCWS1_KW)
    Y1 = PCQ1_KW
    X2 = sm.add_constant(PCWS2_KW)
    Y2 = PCQ2_KW
    
    REGR = sm.OLS(Y1,X1)
    regr = REGR.fit()
    rsquare1 = regr.rsquared
    p1 = regr.params
    a1 = -p1[1] #regression slope
    #
    REGR = sm.OLS(Y2,X2)
    regr = REGR.fit()
    rsquare2 = regr.rsquared
    p2 = regr.params
    a2 = -p2[1]
    
    #######################################
    # For testing only
    bin_x_wide = np.arange(-0.004,0.006,0.002)/4
    bin_y_wide = bin_x_wide
    fig,ax = plt.subplots(1,2,figsize=(5.5, 2.4),dpi=600)
    for i in range(0,2):
        plt.subplot(1,2,i+1)
        plt.subplots_adjust(left=0.14,right=0.97,top=0.9,bottom=0.13,wspace=0.35)
        if i == 0:
            plt.scatter(X1[:,1], Y1, s=2)
            plt.plot(X1[:,1], p1[1]*X1[:,1]+p1[0],'k')
        else:
            plt.scatter(X2[:,1], Y2, s=2)
            plt.plot(X2[:,1], p2[1]*X2[:,1]+p2[0],'k')
        ax = plt.gca()
        ax.tick_params(width=0.5,bottom=True, top=True, left=True, right=True)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax.tick_params(axis="both", direction="in")
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)
        if i == 0:
            plt.title('1st mode',fontsize=9)
            plt.xlabel('PCW1*Sp',fontsize=9)
            plt.ylabel('PCQ1',fontsize=9)
        else:
            plt.title('2nd mode',fontsize=9)
            plt.xlabel('PCW2*Sp',fontsize=9)
            plt.ylabel('PCQ2',fontsize=9)            
        plt.xticks(bin_x_wide,fontsize=8)
        plt.xlim([-0.005/5,0.005/5])
        plt.yticks(bin_y_wide,fontsize=8)
        plt.ylim([-0.005/5,0.005/5])
        #plt.savefig(figdir+'scatter_PCQ_PCWS.png',dpi=600)
    plt.show()
    plt.close()
    ###############################################################
    
    # Calculate equivalent depth
    He1 = N2_1*(1-a1)/((mm1**2+1/(4*Hs**2))*g)
    He2 = N2_2*(1-a2)/((mm2**2+1/(4*Hs**2))*g)

    Cp1 = (g*He1)**0.5
    Cp2 = (g*He2)**0.5
    
    return Cp1, Cp2, He1, He2, Lz1, Lz2, N2_1, N2_2, a1, a2

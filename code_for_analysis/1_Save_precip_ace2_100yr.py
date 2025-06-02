#############################################
# Goal of this code is to save precip data from ACE2
# From 100 years of data (10 members of 2001, 2002...2010)
# This code is modified from the original code: Save_precip_ace2.py
# 2025.2.10
# Mu-Ting Chien
#########################################
# Import pacakges
import numpy as np
import xarray as xr
import os

# Set constants
d2s = 86400

# Set path for figure
expname = 'ace2' 
imem = 0 # Just choose 1 ensemble member
mem_str = f"{imem+1:02d}" 

DIR = '/barnes-engr-scratch1/c832572266/'
fig_dir = DIR + 'figure/ace2_fig/' # Reomte direcotry for figures 
os.makedirs(fig_dir,exist_ok=True) 
file_name = 'autoregressive_predictions.nc'

sub_dir = list(['2001','2002','2003','2004','2005','2006','2007','2008','2009','2010'])
nsub    = np.size(sub_dir)

# Load precip 

for i in range(0, nsub):
    # Load yr 1-10 data
    file_dir = DIR + 'data_output/ace2/ace2_output/repeat_2001-2010/'+sub_dir[i]+'/'
    ds     = xr.open_dataset(file_dir + file_name) #(time, ensemble_member, lat, lon)

    # Find the index of the nearest latitudes to -15 and 15
    lat_15S = ds.lat.sel(lat=-15, method="nearest")
    lat_15N = ds.lat.sel(lat=15, method="nearest")

    # Select data between these latitudes
    ds = ds.sel(lat=slice(lat_15S, lat_15N))
    time = ds['time'][:] 
    lat_15SN = ds['lat'][:]
    
    nt = np.size(time)

    # Load variables: PRECIP 
    #   Precipitaiton at surface (original unit: kg/m2/s), *d2s will change unit into mm/day
    PRECIP = ds['PRATEsfc'][:,:,:,:] 


    mem  = ds['sample']
    lat  = ds['lat']
    lon  = ds['lon']
    nmem = np.size(mem)
    nlat = np.size(lat)
    nlon = np.size(lon)
    print('Finish loading precip')

    #######################################
    # Calculate meridional average of precip
    ########################################
    # Convert latitude to radians for cosine weighting
    lat_radians = np.deg2rad(lat_15SN)
    # Create weights based on cos(latitude)
    weights = np.cos(lat_radians)

    PRECIP_15SN_lat_avg     = PRECIP.weighted(weights).mean(dim='lat')

    #############################
    # Save precip output
    ############################
    os.makedirs(file_dir, exist_ok=True) 

    ds = xr.Dataset({
        "PRECIP":PRECIP,
        "PRECIP_15SN_lat_avg":PRECIP_15SN_lat_avg,
        "lat_15SN":lat_15SN,
    })
    ds.to_netcdf(file_dir + "PRECIP_"+expname+"_"+sub_dir[i]+"_10yr.nc")
    print('Finish saving precip')
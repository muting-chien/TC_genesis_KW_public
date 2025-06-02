####################
# Calculate momentum forcing F = -( d(u'u')/dx + d(u'v')/dy + d(u'w')/dp )
# Input data: 3-hourly output
# Mu-Ting Chien
# 2023.9.24
# ###################
import numpy as np
from numpy import dtype
from netCDF4 import Dataset
import glob
import sys
sys.path.append('/glade/work/muting/function/')
import mjo_mean_state_diagnostics as MJO
import KW_diagnostics as KW
import os

DIR = '/glade/work/muting/'
DIR_in = '/glade/derecho/scratch/muting/FROM_CHEYENNE/'
CASENAME_LIST2 = list(['SST_AQP3_Qobs_27_-4K',\
                       'SST_AQP3_Qobs_27',\
                       'SST_AQP3_Qobs_27_4K'])
CASENAME_SHORT_LIST = list(['-4K','CTL','4K'])
dir_out = DIR+'KW/'
latmin = -45
latmax = 45
vname = list(['U','V','OMEGA'])
Fname = list(['duaua_dx','duava_dy','duawa_dp'])

# Info for loading 3-hourly output of CESM simulation 
nmon = 12
iyr_max = 12 #12 #5 #5 # End with year 5
iyr_min = 2 #2 #Start with year 3
nt_s = 31


for icase in range(0,3):
    CASENAME = CASENAME_LIST2[icase]+'_3h_20y'
    CASENAME_SHORT = CASENAME_SHORT_LIST[icase]+'_20y_3h_20y'
    print(CASENAME)
    file_dir = DIR_in+'work_output_pressure_coord_full/'+CASENAME+'/'
    
    figdir = DIR+'KW/figure/'+CASENAME+'/'
    
    
    ###################
    # load CESM 3-hourly output
    for iyr in range(iyr_min,iyr_max):
        yr_str = str(iyr+1)
        print(yr_str)
        nfile_skip = KW.find_nfile_skip(CASENAME, CASENAME_SHORT, iyr, iyr)
    
        # (1) Load u, v, omega
        for iv in range(0,3):
            if iv == 0:
                vname = list(['U'])
                u, um, time, plev, lon, lat = KW.load_3D_data_as_1variable(CASENAME, CASENAME_SHORT, vname, iyr, iyr, latmax, nfile_skip, kw_proj=0)
                if np.sum(np.isnan(u))!=0:
                    print('Caution: nan in u')
                else:
                    print( 'U min and max:', np.min(u), np.max(u) )
                del um
            elif iv == 1:
                vname = list(['V'])
                v, vm, time, plev, lon, lat = KW.load_3D_data_as_1variable(CASENAME, CASENAME_SHORT, vname, iyr, iyr, latmax, nfile_skip, kw_proj=0)
                del vm
                if np.sum(np.isnan(v))!=0:
                    print('Caution: nan in v')
                else:
                    print( 'V min and max:', np.min(v), np.max(v) )
            elif iv == 2:
                vname = list(['OMEGA'])
                w, wm, time, plev, lon, lat = KW.load_3D_data_as_1variable(CASENAME, CASENAME_SHORT, vname, iyr, iyr, latmax, nfile_skip, kw_proj=0)
                del wm
                if np.sum(np.isnan(w))!=0:
                    print('Caution: nan in w')
                else:
                    print( 'W min and max:', np.min(w), np.max(w) )
                    
            nt   = np.size(time)
            nlev = np.size(plev)
            nlat = np.size(lat)
            nlon = np.size(lon)
            print('start calculating F')
                
            ##################################
            # Calculate F
            # F'= - (d(u'u')/dx + d(u'v')/dy + d(u'w')/dp) 
            # equivalent to F' = du'/dt + u'du/dx + v'du/dy + w'du/dp + udu'/dx + vdu'/dy + wdu'/dp -fv' + dphi'/dx
            ##############################
            if iv == 0:
                u = MJO.remove_10d_from_3hr_data(u)
                duaua_dx = np.empty([nt,nlev-2,nlat-2,nlon])
                dx_temp = 2*2.5*111*1000*np.cos(lat[1:-1]*2*np.pi/360) #(m)
                dx = np.tile(dx_temp,(nt,nlev-2,nlon,1))
                del dx_temp
                dx = np.transpose(dx,(0,1,3,2))
                duaua_dx[:,:,:,1:-1]   = ( u[:,1:-1,1:-1,2:]**2-u[:,1:-1,1:-1,:nlon-2]**2)
                duaua_dx[:,:,:,0]      = ( u[:,1:-1,1:-1,1]**2-u[:,1:-1,1:-1,-1]**2)
                duaua_dx[:,:,:,nlon-1] = ( u[:,1:-1,1:-1,0]**2-u[:,1:-1,1:-1,nlon-2]**2)
                duaua_dx = duaua_dx/dx #(nt,nlev-2,nlat-2,nlon)
                duaua_dx = KW.KW_meridional_projection(duaua_dx, lat[1:-1], 1)
                
                del dx
                print('d(ua^2)/dx')
            
            elif iv == 1:
                v = MJO.remove_10d_from_3hr_data(v)
                duava_dy = u[:,1:-1,2:,:]*v[:,1:-1,2:,:]-u[:,1:-1,:nlat-2,:]*v[:,1:-1,:nlat-2,:]
                dy = (lat[2:]-lat[:nlat-2])*111*1000 #m
                for ilat in range(0,nlat-2):
                    duava_dy[:,:,ilat,:] = duava_dy[:,:,ilat,:]/dy[ilat]
                del v, dy
                duava_dy = KW.KW_meridional_projection(duava_dy, lat[1:-1], 1)
                print('d(ua*va)/dy')
                
            elif iv == 2:
                w = MJO.remove_10d_from_3hr_data(w)
                dp = (plev[2]-plev[0])*100
                duawa_dp = ( u[:,2:,1:-1,:]*w[:,2:,1:-1,:]-u[:,:nlev-2,1:-1,:]*w[:,:nlev-2,1:-1,:] )/dp
                duawa_dp = KW.KW_meridional_projection(duawa_dp, lat[1:-1], 1)
                del w, dp
                print('d(ua*wa)/dp')
        
            if iv == 0:
                print('Max and min of d(uaua)/dx:',np.max(-duaua_dx),np.min(-duaua_dx))
                F = -duaua_dx
                del duaua_dx
            elif iv == 1:  
                print('Max and min of d(uava)/dy:',np.max(-duava_dy),np.min(-duava_dy))
                F = F-duava_dy
                del duava_dy
            elif iv == 2: 
                print('Max and min of d(uawa)/dp:',np.max(-duawa_dp),np.min(-duawa_dp))
                F = F-duawa_dp
                del duawa_dp
        
        ua = KW.KW_meridional_projection(u[:,1:-1,1:-1,:], lat[1:-1], 0)
        del u
        Fa = F-np.transpose( np.tile(np.nanmean(np.nanmean(F,2),0),(nt,nlon,1)), (0,2,1)) 
        del F
        
        #######################
        # Save F (nc)
        os.makedirs(dir_out+'output_data/'+CASENAME+'/F/', exist_ok=True)
        output = dir_out+'output_data/'+CASENAME+'/F/F_eddy_momentum_forcing_'+yr_str+'.nc'
        ncout = Dataset(output, 'w', format='NETCDF4')

        # define axis size
        ncout.createDimension('plev', nlev-2)
        ncout.createDimension('lon',  nlon)
        ncout.createDimension('time', nt)

        # create plev axis
        plev3 = ncout.createVariable('plev', dtype('double').char, ('plev'))
        plev3.standard_name = 'plev'
        plev3.long_name = 'pressure level'    
        plev3.units = 'hPa'
        plev3.axis = 'Y'
        # create lon axis
        lon3 = ncout.createVariable('lon', dtype('double').char, ('lon'))
        lon3.long_name = 'longitude'
        lon3.units = 'deg'
        lon3.axis = 'lon'
        # create time axis
        time3 = ncout.createVariable('time', dtype('double').char, ('time'))
        time3.long_name = 'time'
        time3.units = 'yyyymmdd'
        time3.calendar = 'standard'
        time3.axis = 'T'
        
        F2_out = ncout.createVariable('Fa', dtype('double').char, ('time','plev','lon'))
        F2_out.long_name = 'eddy momentum forcing anomaly = -d(ua*ua)/dx -d(ua*va)/dy - d(ua*wa)/dp '
        F2_out.units = 'm/s^2'  
        
        u_out = ncout.createVariable('ua', dtype('double').char, ('time','plev','lon'))
        u_out.long_name = 'zonal wind anomaly'
        u_out.units = 'm/s'  

        lon3[:] = lon[:]
        plev3[:] = plev[1:-1]
        time3[:] = time[:]

        if np.sum(np.isnan(ua)==1)!=0:
            print('Caution: nan in Fa, located in', np.argwhere(np.isnan(Fa)==1))
        else:
            print('Max and min of u:',np.max(ua),np.min(ua))
        u_out[:] = ua[:]
        del ua

        if np.sum(np.isnan(Fa)==1)!=0:
            print('Caution: nan in Fa, located in', np.argwhere(np.isnan(Fa)==1))
        else:
            print('Max and min of F:',np.max(Fa),np.min(Fa))
        
        F2_out[:] = Fa[:]
        
        
        print('finish saving')
        ncout.close()
        del Fa
       

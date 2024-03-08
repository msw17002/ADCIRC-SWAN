#!/home/msw17002/anaconda3/bin/python

import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import pandas as pd
import numpy as np
import datetime
import argparse
import netCDF4
import glob
import os 

parser = argparse.ArgumentParser()
parser.add_argument('-dto', '--dto', required=True, help='Start of analysis (YYYYmmddHH)')
parser.add_argument('-dte', '--dte', required=True, help='End   of analysis (YYYYmmddHH)')

args = parser.parse_args()
dto  = str(args.dto)
dte  = str(args.dte)

#---query date
dto = datetime.datetime.strptime(dto,"%Y%m%d%H")
dte = datetime.datetime.strptime(dte,"%Y%m%d%H")

#-incriment for processing
inc = datetime.timedelta(hours=1)
#-dates for processing fields
dates = np.arange(dto, dte+inc, inc).astype(datetime.datetime)
#-generate output name
output = "fort-22.nc"
#-groups for processing
groups = []
#-path to nc files 
path = "./data/"
#-path to expected era5 file
era5 = path + "ERA5-" + dto.strftime("%Y%m%d%H-") + dte.strftime("%Y%m%d%H.nc")
#-does ecmwf reanalysis exist?
if os.path.exists(era5):   groups.append("ERA5")
#-does rap analysis exist? 
if len(glob.glob(path+"*RAP*"))>0: groups.append("RAP")
#does hrrr analysis exist?
if len(glob.glob(path+"*HRRR*"))>0: groups.append("HRRR")
#-iterate through groups
for g in range(len(groups)):
    print(groups[g])
    #-preallocate required fields
    data  = [['time'],['u10'],['v10'],['slp']]
    #=====================================================================================
    #====PROCESS ONLY ERA5===
    if ((g==0) and any(np.asarray(groups)=="ERA5")):
        #-read netcdf file 
        nc    = netCDF4.Dataset(era5)
        ##-create an array of datetimes
        #times = np.asarray(nc['initial_time0_hours'][:])
        times = str(nc['initial_time0'][:].tobytes())[2:-2].split(")")
        tquer = []
        for t in times:
            ##-base time 
            #bt = datetime.datetime.strptime(nc['time'].units,"hours since %Y-%m-%d %H:%M:%S")
            ##-combine times
            #tquer.append(bt+datetime.timedelta(hours=float(t)))
            bt = datetime.datetime.strptime(t.replace("(",""),"%m/%d/%Y %H:%M")
            tquer.append(bt)
        #-main appending script
        for d in range(len(dates)):
            print("    Processing "+groups[g]+" for "+dates[d].strftime("%Y-%m-%dT%H")+"!!!")
            #-get coordinates
            if d==0:
                ##-1d
                #lon1d = nc['lon'][:]
                #lat1d = nc['lat'][:]
                lon1d = nc['g0_lon_2'][:]
                lat1d = nc['g0_lat_1'][:]
                #-2d
                lon2d,lat2d = np.meshgrid(lon1d,lat1d)
            #-get time entry 
            rec   = np.where(np.asarray(tquer)==dates[d])[0][0]
            #-get query fields
            #u10   = np.asarray(nc['var165'][rec,:,:])
            #v10   = np.asarray(nc['var166'][rec,:,:])
            #slp   = np.asarray(nc['var151'][rec,:,:])*0.01 #Pa to mb
            u10   = np.asarray(nc['10U_GDS0_SFC'][rec,:,:])
            v10   = np.asarray(nc['10V_GDS0_SFC'][rec,:,:])
            slp   = np.asarray(nc['MSL_GDS0_SFC'][rec,:,:])*0.01 #Pa to mb
            time  = dates[d]
            #-append to tuple
            data[0].append(time)
            data[1].append(u10)
            data[2].append(v10)
            data[3].append(slp)
        #-close nc file 
        nc.close()
    #=====================================================================================
    #====PROCESS EITHER RAP OR HRRR===
    else:
        for d in range(len(dates)):
            print("    Processing "+groups[g]+" for "+dates[d].strftime("%Y-%m-%dT%H")+"!!!")
            #-read netcdf file 
            nc = netCDF4.Dataset(path+groups[g]+"-"+dates[d].strftime("%Y%m%d%H.nc"))
            #-get coordinates
            if d==0:
                lon2d = nc['longitude'][::-1,:]-360
                lat2d = nc['latitude' ][::-1,:]
            #-get query fields
            u10   = np.asarray(nc['UGRD_10maboveground'][0,::-1,:])
            v10   = np.asarray(nc['VGRD_10maboveground'][0,::-1,:])
            slp   = np.asarray(nc['MSLMA_meansealevel' ][0,::-1,:])*0.01 #Pa to mb
            time  = datetime.datetime.strptime(nc['time'].units,"seconds since %Y-%m-%d %H:%M:00.0 0:00")+\
                    datetime.timedelta(hours=float(nc['time'][0].data)/(3600))
            #-append to tuple
            data[0].append(time)
            data[1].append(u10)
            data[2].append(v10)
            data[3].append(slp)
            #-close nc file 
            nc.close()
    #=====================================================================================
    #-create an xarray dataset 
    ds = xr.Dataset(data_vars=dict(U10 =(["time","yi", "xi"], np.asarray(data[1][1:])),
                                   V10 =(["time","yi", "xi"], np.asarray(data[2][1:])),
                                   PSFC=(["time","yi", "xi"], np.asarray(data[3][1:])),
                                   time=data[0][1:],
                                   reference_time=data[0][1],),
                    coords=dict(lon=(["yi", "xi"], lon2d),
                                lat=(["yi", "xi"], lat2d),))
    #-set coordinates 
    ds = ds.set_coords(("time", "lat", "lon"))
    #-set coordinate system
    ds["lon"].attrs = {"units":"degrees_east",
                       "standard_name":"longitude",
                       "axis": "xi",
                       "coordinates":"time lat lon",}
    ds["lat"].attrs = {"units":"degrees_north",
                       "standard_name":"latitude",
                       "axis":"yi",
                       "coordinates":"time lat lon",}
    #-set units and coordinates of fields
    ds["time"].attrs = {"axis":      "T", "coordinates": "time"        }
    ds["U10" ].attrs = {"units": "m s-1", "coordinates": "time lat lon"}
    ds["V10" ].attrs = {"units": "m s-1", "coordinates": "time lat lon"}
    ds["PSFC"].attrs = {"units":    "mb", "coordinates": "time lat lon"}
    #-not sure what's going on here
    ds.encoding = {}
    #-set rank and naming convension
    ds.attrs["rank"] = int(g+1)
    ds.time.encoding = {"units": "minutes since 1990-01-01 01:00:00",
                        "dtype": int,}
    #-write fields to netcdf file
    if g==0: ds[["U10", "V10", "PSFC"]].to_netcdf(output, group=groups[g], mode="w")
    else:    ds[["U10", "V10", "PSFC"]].to_netcdf(output, group=groups[g], mode="a")
#-append attributes to nc file 
with netCDF4.Dataset(output, "a") as nc:
    nc.setncattr("group_order", " ".join(groups))
    nc.setncattr("institution", "Oceanweather Inc. (OWI)")
    nc.setncattr("conventions", "CF-1.6 OWI-NWS13")
#-make sure the file syntax is correct
nc = netCDF4.Dataset(output)
for g in groups:
    print("===================================================================")
    print("==="+g+"===")
    print(np.asarray(nc[g]['lat'][:,:]))
    print("-------------------------------------------------------------------")
    print(np.asarray(nc[g]['lon'][:,:]))

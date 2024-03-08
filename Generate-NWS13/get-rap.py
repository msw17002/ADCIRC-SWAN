#!/home/msw17002/anaconda3/bin/python

import requests
import datetime
import argparse
import time
import os

#To Run: ./get-rap.py -dto 2012102600 -dte 2012103012
#Need to configure compile wgrib2
parser = argparse.ArgumentParser()
parser.add_argument('-dto', '--dto', required=True, help='Start of analysis (YYYYmmddHH)')
parser.add_argument('-dte', '--dte', required=True, help='End   of analysis (YYYYmmddHH)')

args = parser.parse_args()
dto  = str(args.dto)
dte  = str(args.dte)

#---query date
dto = datetime.datetime.strptime(dto,"%Y%m%d%H")
dte = datetime.datetime.strptime(dte,"%Y%m%d%H")

#-base path for data
base = "https://www.ncei.noaa.gov/data/rapid-refresh/access/"
#-constant datetime threshold
k    = datetime.datetime.strptime("2020051508","%Y%m%d%H")

#---path to wgrib2
wgrib2 = "/path/to/wgrib2"

while dto!=dte+datetime.timedelta(hours=1):
    #-file to be downloaded
    if k<dto: url = base + dto.strftime("rap-130-13km/analysis/%Y%m/%Y%m%d/rap_130_%Y%m%d_%H00_000.grb2") 
    else:     url = base + dto.strftime("historical/analysis/%Y%m/%Y%m%d/rap_130_%Y%m%d_%H00_000.grb2")  
    #url = base + dto.strftime("historical/analysis/%Y%m/%Y%m%d/ruc2anl_252_%Y%m%d_%H00_000.grb") 
    #url = base + dto.strftime("historical/analysis/%Y%m/%Y%m%d/rap_252_%Y%m%d_%H00_000.grb2")
    #-output file name
    out = dto.strftime("./data/RAP-%Y%m%d%H.grb2")
    #-notification
    print("Working on: " + out + "!")
    if os.path.exists(out): pass
    else:
        #-download file
        response = requests.get(url)
        with open(out, "wb") as f:
            f.write(response.content)
        #-give server rest
        time.sleep(5)
    #-subset for only 10m U/V and MSLP 
    if os.path.exists(out+".subset"): pass
    else:                             os.system(wgrib2+' '+out+' -match ":MSLMA|:UGRD:10 m above ground|:VGRD:10 m above ground" -grib '+out+'.subset')
    #-convert to a netcdf4 file
    if os.path.exists(out.replace(".grb2",".nc")): pass
    else:
        print("Mike:",wgrib2+' '+out+".subset"+" -netcdf "+out.replace(".grb2",".nc"))
        #-convert
        os.system(wgrib2+' '+out+".subset"+" -netcdf "+out.replace(".grb2",".nc"))
        #-make sure files are gone
        if os.path.exists(out.replace(".grb2",".nc")):
            if os.path.exists(out):           os.remove(out)
            if os.path.exists(out+".subset"): os.remove(out+".subset")
    #-update datetime 
    dto += datetime.timedelta(hours=1)

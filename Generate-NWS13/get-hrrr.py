#!/home/msw17002/anaconda3/envs/herbie/bin/python

from herbie import Herbie
import pandas as pd
import numpy as np
import datetime 
import argparse
import glob 
import time
import os 

#---output to data
path = "./data/"
if os.path.exists(path): pass
else:                    os.mkdir(path)

#---temp directory for downloading
diro = "./temp/"
if os.path.exists(diro): pass
else:                    os.mkdir(diro)

#To Run: ./get-hrrr.py -dto 2012102600 -dte 2012103012
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

#---path to wgrib2 
wgrib2 = "/path/to/wgrib2"

#---main script
while dto!=dte+datetime.timedelta(hours=1):
    print("Generating: "+dto.strftime("hrrr-%Y%m%d%H.grib2"))
    #-expected file on local drive
    out  = path + dto.strftime("HRRR-%Y%m%d%H.grib2")
    #-download file
    if os.path.exists(out): pass
    else:
        H = Herbie(dto.strftime("%Y-%m-%d %H:00"), model="hrrr", product="sfc", fxx=0)
        H.download(save_dir=diro) #-download the entire file
        time.sleep(5)
        #-rename file
        ls = glob.glob(diro+"/hrrr/"+dto.strftime("%Y%m%d")+"/*")
        for l in ls:
            #-input file
            inn = l
            #-change file name
            os.system("mv "+inn+" "+out)
        if os.path.exists(diro+"/hrrr/"):
            #-remove directory
            os.system("rm -rf "+diro+"/hrrr/")
            #-sleep for several seconds
            time.sleep(2)
    #-subset for only 10m U/V and MSLP
    if os.path.exists(out+".subset"): pass
    else:                             os.system(wgrib2+'  '+out+' -match ":MSLMA|:UGRD:10 m above ground|:VGRD:10 m above ground" -grib '+out+'.subset')
    #-convert to a netcdf4 file
    if os.path.exists(out.replace(".grib2",".nc")): 
        #-make sure files are gone
        if os.path.exists(out):           os.remove(out)
        if os.path.exists(out+".subset"): os.remove(out+".subset")
    else:
        #-convert
        os.system(wgrib2+' '+out+".subset"+" -netcdf "+out.replace(".grib2",".nc"))
        #-make sure files are gone
        if os.path.exists(out):           os.remove(out)
        if os.path.exists(out+".subset"): os.remove(out+".subset")
    #-add datetime 
    dto += datetime.timedelta(hours=1)

#!/home/msw17002/anaconda3/bin/python

import argparse
import datetime
import argparse
import cdsapi

#To Run: ./get-era5.py -dto 2012102600 -dte 2012103012
parser = argparse.ArgumentParser()
parser.add_argument('-dto', '--dto', required=True, help='Start of analysis (YYYYmmddHH)')
parser.add_argument('-dte', '--dte', required=True, help='End   of analysis (YYYYmmddHH)')

args = parser.parse_args()
dto  = str(args.dto)
dte  = str(args.dte)

#---query date
dto = datetime.datetime.strptime(dto,"%Y%m%d%H")
dte = datetime.datetime.strptime(dte,"%Y%m%d%H")

#---don't change this... WPS fails if you subset the ERA5 grid
North =  90
South = -90
East  =  180
West  = -180

#---get extraction dates
bdto = (dto-datetime.timedelta(hours=6)).strftime("%Y%m%d")
bdte = (dte+datetime.timedelta(hours=6)).strftime("%Y%m%d")

#---get surface data first
c = cdsapi.Client()
c.retrieve('reanalysis-era5-single-levels',{'product_type':'reanalysis','format':'grib','variable':[
                '10m_u_component_of_wind','10m_v_component_of_wind','mean_sea_level_pressure'],
        'date':bdto+'/'+bdte,
        'area':str(North)+'/'+str(West)+'/'+str(South)+'/'+str(East),
        'time':['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00',
                '13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00'],
        'grid':"0.25/0.25",},
               './data/ERA5-'+dto.strftime("%Y%m%d%H")+'-'+dte.strftime("%Y%m%d%H")+'.grib')

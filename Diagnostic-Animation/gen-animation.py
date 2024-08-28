#!/var/www/kmz/MikesToolz/ADCIRC/kalpana/Kalpana/bin/python

#===this function plots the max. and min. value onto a projected map
def min_max(zfield,minopt,maxopt):
    #-plot min
    if minopt==1:
        ax.scatter(float(points['LON'][minr]),float(points['LAT'][minr]),s=300,edgecolors='b',
                facecolors='none',transform=crs.PlateCarree(),zorder=502)
        ax.scatter(float(points['LON'][minr]),float(points['LAT'][minr]),marker='x',s=300,facecolors='b',
                transform=crs.PlateCarree(),zorder=502)
    #-plot max
    if maxopt==1:
        ax.scatter(float(points['LON'][maxr]),float(points['LAT'][maxr]),s=300,
                edgecolors='r',facecolors='none',transform=crs.PlateCarree(),zorder=502)
        ax.scatter(float(points['LON'][maxr]),float(points['LAT'][maxr]),marker='x',s=300,facecolors='r',
                transform=crs.PlateCarree(),zorder=502)

#===this function creates a dataframe capable of plotting directional
#   values on a constant grid
def gen_quiv(U,V,Z,N,E,S,W):
    #-et up transformers, EPSG:3857 is metric, same as EPSG:900913
    to_proxy_transformer    = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857')
    to_original_transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857')
    #-reate corners of rectangle to be transformed to a grid
    sw = shapely.geometry.Point((S, W))
    ne = shapely.geometry.Point((N, E))
    #-grid resolution in meters
    stepsize = np.min((dx,dy))/10
    #-project corners to target projection
    transformed_sw = to_proxy_transformer.transform(sw.x, sw.y) # Transform NW point to 3857
    transformed_ne = to_proxy_transformer.transform(ne.x, ne.y) # .. same for SE
    #-iterate over 2D area
    grid = [['x'],['y']]
    x = transformed_sw[0]
    while x < transformed_ne[0]:
        y = transformed_sw[1]
        while y < transformed_ne[1]:
            grid[0].append(x)
            grid[1].append(y)
            y += stepsize
        x += stepsize
    #-to pandas dataframe
    grid = pd.DataFrame(grid).T
    grid.columns = grid.iloc[0]
    grid = grid.drop(0).reset_index(drop=True)
    #-to geopandas dataframe
    grid = geopandas.GeoDataFrame(grid, geometry=geopandas.points_from_xy(grid.x, grid.y), crs="EPSG:3857").to_crs('epsg:4326')
    #-determine which points lay in grid
    grid = geopandas.sjoin_nearest(grid, points, distance_col="distances",lsuffix="left", rsuffix="right")
    #-clean and filter
    grid = grid[['index_right','LON','LAT']].drop_duplicates().reset_index(drop=True)
    #-create a quiver plot
    quiv = [['u'],['v'],['z']]
    for i in range(grid.shape[0]):
        quiv[0].append(U[grid['index_right'][i]])
        quiv[1].append(V[grid['index_right'][i]])
        quiv[2].append(Z[grid['index_right'][i]])
    #-to pandas dataframe
    quiv = pd.DataFrame(quiv).T
    quiv.columns = quiv.iloc[0]
    quiv = quiv.drop(0).reset_index(drop=True)
    #-join jolumns
    quiv = pd.concat([grid, quiv], axis = 1)
    return(quiv)

##===this function calculates u/v components from direction and intensity
#def wind_spddir_to_uv(mag,dir):
#    """
#    calculated the u and v wind components from wind speed and direction
#    Input:
#        wspd: wind speed
#        wdir: wind direction
#    Output:
#        u: u wind component
#        v: v wind component
#    """
#    rad = 4.0*np.arctan(1)/180.
#    u = -mag*np.sin(rad*dir)
#    v = -mag*np.cos(rad*dir)
#    return u,v

def get_uv(mag,drct):
    #-create u/v components
    comU,comV = [],[]
    for i in range(len(drct)):
        #-calculate wave compenents
        if (np.isnan(mag[i]) or np.isnan(drct[i])): u,v = 0,0
        else:                                       u,v = mag[i] * np.cos(drct[i] * (np.pi/180)),mag[i] * np.sin(drct[i] * (np.pi/180))
        #-append to lists
        comU.append(u)
        comV.append(v)
    #-convert to arrays
    comU = np.asarray(comU)
    comV = np.asarray(comV)
    return(comU,comV)

import warnings
warnings.filterwarnings("ignore")

from shapely.geometry import Polygon, Point
import matplotlib.patheffects as pe
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as crs
import matplotlib as mpl
from tqdm import tqdm
import pandas as pd 
import numpy as np
import geopandas 
import datetime
import netCDF4
import sys
import os


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import shapely.geometry
import matplotlib
import pyproj
import glob

path = r"/path/to/adcirc/output/"
unit = "imperial"
vortex = 1 #need fort.22 for this
           #if set to 0, will create domain figures
#---read global shapefil
coast = cfeature.GSHHSFeature(scale="f")

#-preset expected booleans
gh2O,gwv,gslp,gwind = 0,0,0,0

#-do the files exist to generate water level graphics
if (os.path.exists(path + "fort.63.nc") and os.path.exists(path + "fort.64.nc")):
    #-read files
    h2O  = netCDF4.Dataset(path + "fort.63.nc") #zeta
    h2Od = netCDF4.Dataset(path + "fort.64.nc") #u-vel & v-vel
    #-update graphics boolean 
    gh2O = 1
    #-ensure the time dimensions are equal 
    if all(h2O['time'][:].data==h2Od['time'][:].data): pass
    else:                                              sys.exit("Time dimensions aren't equal!!!") 
    #-do observations exist from evaluation script?
    ls = glob.glob(path + "/evaluation/water-levels/text/*xy-*.csv")
    if (len(ls)>0 and os.path.exists(path + "/evaluation/water-levels/meta-water-levels.csv")):
        #-read meta file for coordinates
        meta = pd.read_csv(path + "/evaluation/water-levels/meta-water-levels.csv")
        #-concat csvs 
        for l in range(len(ls)):
            #-read csv 
            csv = pd.read_csv(ls[l])
            if l==0: coops = csv
            else:    coops = pd.concat([coops,csv]).reset_index(drop=True)
        #-join coordinate data 
        coops = coops.merge(meta,on='id')
        #-convert units if needed 
        if unit=='si': 
            #-no conversion
            coops['water-obs'] = coops['observed-water-level-meters']
        else:
            #-convert to feet 
            coops['water-obs'] = coops['observed-water-level-meters']*3.28084
        ##-filter observations 
        #coops = coops[['predicted-datetime-UTC','water-obs','lat','lon']].dropna(subset=['water-obs']).reset_index(drop=True)
        ##-to geopandas dataframe 
        #coops = geopandas.GeoDataFrame(coops, geometry=geopandas.points_from_xy(coops.lon, coops.lat), crs="EPSG:4326")
        ##-set observations boolean
        obsC = 1
    #-convert fields
    if   ((h2O['zeta'].units=='m') and (unit=='si')):     
        #-converted fields
        zeta = 1 #m to m
        wz   = 1 #m/s to m/s
        qvlb = "m/s"
    elif ((h2O['zeta'].units=='m') and (unit=='imperial')): 
        #-converted fields
        zeta = 3.28084 #m to ft
        wz   = 2.23694 #m/s to miles/hour 
        qvlb = "miles/hour"
    elif ((h2O['zeta'].units!='m') and (unit=='imperial')):  
        #-converted fields
        zeta = 1        #ft to ft
        wz   = 0.681818 #ft/s to miles/hour
        qvlb = "miles/hour"
    elif ((h2O['zeta'].units!='m') and (unit=='si')):        
        #-converted fields
        zeta = 1/3.28084 #ft to m
        wz   = 0.3048    #ft/s to m/s
        qvlb = "m/s"
    else: sys.exit("Conversion b/n "+h2O['zeta'].units+" to "+unit+" isn't cataloged in script! Must append conversion!!!")

#-do the files exist to generate wave height graphics
if (os.path.exists(path + "swan_DIR.63.nc") and os.path.exists(path + "swan_HS.63.nc")): 
    #-read files
    wvht = netCDF4.Dataset(path + "swan_HS.63.nc" ) #swan_HS
    wvd  = netCDF4.Dataset(path + "swan_DIR.63.nc") #swan_DIR
    #-ensure the time dimensions are equal
    if all(wvht['time'][:].data==wvd['time'][:].data): pass
    else:                                              sys.exit("Time dimensions aren't equal!!!")
    #-update graphics boolean
    gwv  = 1
    #-do observations exist from evaluation script?
    ls = np.sort(glob.glob(path + "/evaluation/waves/text/*xy-*.csv"))
    if (len(ls)>0 and os.path.exists(path + "/evaluation/waves/meta-waves.csv")):
        #-read meta file for coordinates
        meta = pd.read_csv(path + "/evaluation/waves/meta-waves.csv")
        #-concat csvs 
        for l in range(len(ls)):
            #-read csv 
            csv = pd.read_csv(ls[l])
            if l==0: buoys = csv
            else:    buoys = pd.concat([buoys,csv]).reset_index(drop=True)
        #-join coordinate data 
        buoys = buoys.merge(meta,on='id')
        #-convert units if needed 
        if unit=='si': 
            #-no conversion
            buoys['waves-obs'] = buoys['waves-observed-meters' ]
        else:
            #-convert to feet 
            buoys['waves-obs'] = buoys['waves-observed-meters' ]*3.28084
        #-filter observations 
        buoys = buoys[['lon','lat','waves-obs','join']].dropna(subset=['waves-obs']).reset_index(drop=True)
        #-to geopandas dataframe 
        buoys = geopandas.GeoDataFrame(buoys, geometry=geopandas.points_from_xy(buoys.lon, buoys.lat), crs="EPSG:4326")
        #-add datetime
        buoys['datetime'] = pd.to_datetime(buoys['join'],format="%Y%m%d%H%M")
        #-set observations boolean
        obsB = 1
    #-get and convert to array
    if   ((wvht['swan_HS'].units=='m') and (unit=='si')):        wvc = 1         #meter to meter 
    elif ((wvht['swan_HS'].units=='m') and (unit=='imperial')):  wvc = 3.28084   #meter to foot
    elif ((wvht['swan_HS'].units!='m') and (unit=='imperial')):  wvc = 1         #foot to foot
    elif ((wvht['swan_HS'].units!='m') and (unit=='si')):        wvc = 1/3.28084 #foot to meter
    else: sys.exit("Conversion b/n "+wvht['swan_HS'].units+" to "+unit+" isn't cataloged in script! Must append conversion!!!")

#-do the files exist to generate sea-level-pressure graphics
if os.path.exists(path + "fort.73.nc"): 
    #-read file 
    slp  = netCDF4.Dataset(path + "fort.73.nc")
    #-update graphics boolean 
    gslp = 1
    #-get and convert to array
    if unit=='si': slpc = 98.04139432*100 #"pa"
    else:          slpc = 98.04139432     #'mb'

#-do the files exist to generate wind speed graphics
if os.path.exists(path + "fort.74.nc"):
    #-read files
    wnd  = netCDF4.Dataset(path + "fort.74.nc") #windx,windy
    #-update graphics boolean
    gwind = 1
    #-get and covert to array 
    if   ((wnd['windx'].units=='m s-1') and (unit=='si')):
        #-converted fields
        ww   = 1 #m/s to m/s
    elif ((wnd['windx'].units=='m s-1') and (unit=='imperial')):
        #-converted fields
        ww   = 2.23694 #m/s to miles/hour
    elif ((wnd['windx'].units!='m s-1') and (unit=='imperial')):
        #-converted fields
        ww   = 0.681818 #ft/s to miles/hour
    elif ((wnd['windx'].units!='m s-1') and (unit=='si')):
        #-converted fields
        ww   = 0.3048    #ft/s to m/s

#---define grid using the h2O netcdf file
lon1d   = h2O['x'][:].data
lat1d   = h2O['y'][:].data
#-triangular mesh
npgrid = h2O['element'][:,:].data
lonlat = np.transpose(np.vstack((lon1d,lat1d)))
#-get triangle vorticies. create geopandas dataframe
# of triangular mesh
triangles = []
mesh      = []
for node in tqdm(npgrid):
    #-determine indicies
    i0   = lonlat[int(node[0]-1)]
    i1   = lonlat[int(node[1]-1)]
    i2   = lonlat[int(node[2]-1)]
    #-create a polygon
    mesh.append(Polygon(zip([i0[0],i1[0],i2[0],i0[0]], [i0[1],i1[1],i2[1],i0[1]])))
    #-for contouring
    triangles.append([int(node[0]-1),int(node[1]-1),int(node[2]-1)])
#-convert to pandas dataframe
mesh = pd.DataFrame(mesh)
mesh.columns   = ['geometry']
#-convert to geopandas dataframe
mesh = geopandas.GeoDataFrame(mesh, crs="EPSG:4326", geometry='geometry')
#-get lon,lat coordinates of triangle center. convert to arrays
elements  = np.asarray(triangles)
triang    = tri.Triangulation(lon1d,lat1d)
#-create geopandas dataframe of grid points
points = pd.DataFrame(lonlat)
points.columns = ['LON','LAT']
points = geopandas.GeoDataFrame(points, geometry=geopandas.points_from_xy(points.LON, points.LAT), crs="EPSG:4326")

if vortex==1:
    #-read and parse fort.22 file
    if os.path.exists(path + "/fort.22"): pass
    else:                                 sys.exit(path+"/fort.22 does not exist!")
    #-read
    with open(path+"fort.22") as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    #-process
    xyptw = [[],[],[],[],[]]
    for row in lines:
        latq = float((    row.split(",")[6][0:3].strip()+"."+row.split(",")[6][3:].strip()).replace("N",""))
        lonq = float(("-"+row.split(",")[7][0:4].strip()+"."+row.split(",")[7][4:].strip()).replace("W",""))
        xyptw[0].append(lonq)
        xyptw[1].append(latq)
        xyptw[2].append(int(row.split(",")[9].strip()))
        xyptw[3].append(datetime.datetime.strptime(row.split(",")[2].strip(),"%Y%m%d%H"))
        xyptw[4].append(int(row.split(",")[8].strip()))
    #-Turn into a pandas df
    xyptw = pd.DataFrame(np.asarray(xyptw).transpose())
    xyptw.columns = ['LON','LAT','PRESS_mb','DATETIME','WIND_kt']
    xyptw = xyptw.drop_duplicates().reset_index(drop=True)
    #-reset bbox boolean just to be safe 
    bbox  = None


#-only proceed if all files are available
if (gh2O+gwv+gslp+gwind)==4:
    #-ensure that the images directory exists
    if os.path.exists("./images/"): pass
    else:                           os.mkdir("./images/")
    #-inform
    print("All files are available!")
    #-ensure that they all have the same dimensions
    if (all(h2O['time'][:].data==wvht['time'][:].data) and \
        all(h2O['time'][:].data==slp['time'][:].data) and \
        all(h2O['time'][:].data==wnd['time'][:].data)): 
        #-inform
        print("Time dimensions are equal!")
        #-create time array 
        base  = datetime.datetime.strptime(h2O['time'].units.split("!")[0].split("UTC")[0].strip(),"seconds since %Y-%m-%d %H:%M:%S")
        dts   = h2O['time'][:].data 
        #-generate array
        dates = [(base + datetime.timedelta(hours=d/(60*60))) for d in dts]
        #-conditionally filter dates array
        if vortex==1:
            #-determine max and min dates of xyptw 
            mind,maxd = np.min(xyptw['DATETIME']),np.max(xyptw['DATETIME'])
            #-filter dates by mind and maxd
            log   = np.asarray((mind<=np.asarray(dates)) & (maxd>=np.asarray(dates)))
            dates = list(np.asarray(dates)[log])
        #-generate figures
        for d in range(len(dates)):
            #-determine bounds of image if necessary
            #-this option will zoom in on the hurricane center
            if vortex==1:
                #---set boolean 
                init = 1
                #---current image date
                dtin = dates[d]
                #---interpolate observed data
                dtm  = []
                dtp  = []
                dto  = []
                for dtq in xyptw['DATETIME']:
                    dtS = ((dtq-dtin).seconds)
                    dtD = ((dtq-dtin).days*24*60*60)
                    dt = dtS+dtD
                    if dt==0: dto.append(0)
                    elif dt>0:
                        dtp.append(abs(dt))
                        dtm.append(np.nan)
                        dto.append(np.nan)
                    else:
                        dtp.append(np.nan)
                        dtm.append(abs(dt))
                        dto.append(np.nan)
                #-determine index
                if any(np.asarray(dto)==0):
                    idx = np.nanargmin(dto)
                    #-no interpolation
                    loni = xyptw['LON'     ][idx]
                    lati = xyptw['LAT'     ][idx]
                    wind = xyptw['WIND_kt' ][idx]
                    pres = xyptw['PRESS_mb'][idx]
                    if init==1:
                        plotxo = list(xyptw['LON'][:idx])
                        plotyo = list(xyptw['LAT'][:idx])
                        init   = 0
                else:
                    #-determine index
                    idxm = np.nanargmin(dtm)
                    idxp = np.nanargmin(dtp)
                    #-linear interpolation
                    dtmr = 1-(dtm[idxm]/(dtm[idxm]+dtp[idxp]))
                    dtpr = 1-(dtp[idxp]/(dtm[idxm]+dtp[idxp]))
                    loni = xyptw['LON'     ][idxm]*dtmr+xyptw['LON'     ][idxp]*dtpr
                    lati = xyptw['LAT'     ][idxm]*dtmr+xyptw['LAT'     ][idxp]*dtpr
                    wind = xyptw['WIND_kt' ][idxm]*dtmr+xyptw['WIND_kt' ][idxp]*dtpr
                    pres = xyptw['PRESS_mb'][idxm]*dtmr+xyptw['PRESS_mb'][idxp]*dtpr
                    if init==1:
                        plotxo = list(xyptw['LON'][:idxp])
                        plotyo = list(xyptw['LAT'][:idxp])
                        init   = 0
                plotxo.append(loni)
                plotyo.append(lati)
                #-calculate N,E,S,W bounds of plot 
                point = pd.DataFrame([[loni],[lati]]).T
                point.columns = ['lon','lat']
                point = geopandas.GeoDataFrame(point, geometry=geopandas.points_from_xy(point.lon, point.lat), crs="EPSG:4326")
                #-add buffer around point 
                if unit=='imperial':
                    #-domain extent in miles: 200
                    W,S,E,N = np.asarray(point.to_crs("epsg:3857").buffer(321869).to_crs("epsg:4326").bounds)[0]
                    ##-domain extent in miles: 200
                    #W,S,E,N = np.min(points['LON']),np.min(points['LAT']),np.max(points['LON']),np.max(points['LAT'])
                elif unit=='si':
                    #-domain extent in km: 300 
                    W,S,E,N = np.asarray(point.to_crs("epsg:3857").buffer(300000).to_crs("epsg:4326").bounds)[0]
                else: sys.exit("Unknown system!")
                #-create points to calculate distances
                PNW = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[Point( W,N )]).to_crs('EPSG:3857')
                PSW = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[Point( W,S )]).to_crs('EPSG:3857')
                PSE = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[Point( E,S )]).to_crs('EPSG:3857')
                #-determine horizontal distance
                dx = PNW.distance(PSE)[0]
                #-determine the latitudinal distance
                dy = PSE.distance(PSW)[0]
                #-generate polygon of bounds
                geom   = Polygon(zip([W,E,E,W,W], [N,N,S,S,N]))
                #pol    = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geom])
                #-determine which triangles intersect pol
                geolog = mesh.intersects(geom)
                #-calculate grid for quiver
            else:
                bbox = None
                N,E,S,W = np.max(points['LAT']),np.max(points['LON']),np.min(points['LAT']),np.min(points['LON'])
                N,E,S,W = 41.5,-73,40.3,-74.3 
                #-create points to calculate distances
                PNW = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[Point( W,N )]).to_crs('EPSG:3857')
                PSW = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[Point( W,S )]).to_crs('EPSG:3857')
                PSE = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[Point( E,S )]).to_crs('EPSG:3857')
                #-determine horizontal distance
                dx = PNW.distance(PSE)[0]
                #-determine the latitudinal distance
                dy = PSE.distance(PSW)[0]

            #-main script
            for o in range(2):
                #-met forcing figure from adcirc 
                if ((o==0) and (os.path.exists( (path + "/images/"+dates[d].strftime("AA-%Y%m%d%H%M%S.png")).replace("//","/").replace("\\","/") )==False)):
                    #-output image 
                    out = "./images/"+dates[d].strftime("AA-%Y%m%d%H%M%S.png")
                    print("Generating Figure: "+out)
                    #-subset fields
                    pit  = slp['pressure'][d,:].data*slpc
                    uit  = np.where( wnd['windx'][d,:].data!=-99999,wnd['windx'][d,:].data,0 ) * ww
                    vit  = np.where( wnd['windy'][d,:].data!=-99999,wnd['windy'][d,:].data,0 ) * ww
                    wsit = ( ( uit**2 + vit**2 )**0.5 )
                    #-create figure
                    fig = plt.figure(figsize=(16, 9))
                    for f in range(2):
                        #-determine min and max of the file
                        if f==0:
                            if vortex==1:
                                #-subset
                                log       = points.intersects(geom) & (wnd['windx'][d,:].data!=-99999) & \
                                             (wnd['windy'][d,:].data!=-99999)
                                maxs,_ = str(np.round(np.nanmax(wsit[log]),1)),str(np.round(np.nanmin(wsit[log]),1))
                                #-determine row of min and max
                                maxr = np.nanargmax(np.where(log==True,wsit,np.nan))
                            else:
                                #-subset
                                log       = (wnd['windx'][d,:].data!=-99999) & (wnd['windy'][d,:].data!=-99999)
                                maxs,_ = str(np.round(np.nanmax(wsit[log]),1)),str(np.round(np.nanmin(wsit[log]),1))
                                #-determine row of min and max
                                maxr = np.nanargmax(np.where(log==True,wsit,np.nan))
                        else:
                            if vortex==1:
                                #-subset
                                log       = points.intersects(geom) & (slp['pressure'][d,:].data!=-99999)
                                _,mins = str(np.round(np.nanmax(pit[log]),1)),str(np.round(np.nanmin(pit[log]),1))
                                #-determine row of min and max
                                minr = np.nanargmin(np.where(log==True,pit,np.nan))
                            else:
                                #-subset
                                log       = slp['pressure'][d,:].data!=-99999
                                _,mins = str(np.round(np.nanmax(pit[log]),1)),str(np.round(np.nanmin(pit[log]),1))
                                #-determine row of min and max
                                minr = np.nanargmin(np.where(log==True,pit,np.nan))
                        #-create axis
                        ax = fig.add_subplot(1, 2, f+1, projection=crs.PlateCarree())
                        ax.set_extent([W,E,S,N],crs=crs.PlateCarree())
                        #-add mapping resources
                        ax.add_feature(coast,edgecolor='black',lw=2.0)
                        #-create triangulation
                        if ((bbox!=None) or (vortex==1)):
                            #-masking by geom
                            triang = tri.Triangulation(lon1d,lat1d,triangles=triangles,mask=~geolog)
                        else:
                            #-no masking
                            triang = tri.Triangulation(lon1d,lat1d,triangles=triangles)
                        #-determine which field to plot 
                        if f==0:
                            #-define title
                            title = "Water Surace Wind Velocity and Wind Direction""\n"+"Valid:"+dates[d].strftime("%Y-%m-%dT%H:%M:%S UTC")
                            #-wind speed
                            if unit=='imperial':
                                #-units label
                                units = "miles/hour"
                                #-color levels
                                clev  = np.arange(0,100+2,2)
                            else:
                                #-units label
                                units = "meters/second"
                                #-color levels
                                clev  = np.arange(0,40+1,1)
                            #-define label
                            label   = "wind speed ("+units+")"
                            #-generate colorbar
                            colors1 = plt.cm.Greys(np.linspace(0., 0.5, 5))   #5,10,15,20,25,30,35
                            colors2 = plt.cm.winter(np.linspace(0.5, 1, 10)) #Blues plt.cm.Greens_r(np.linspace(0, 1, 128))
                            colors3 = plt.cm.YlOrRd(np.linspace(0., 1, 10))
                            colors4 = plt.cm.gnuplot2(np.linspace(0.,0.9, 15))
                            colors  = np.vstack((colors1, colors2, colors3, colors4))
                            cmap    = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
                            norm    = mpl.colors.BoundaryNorm(clev, cmap.N, extend='max')
                            #-set extend option for colorbar
                            opt     = 'max'
                            #-plot shaded image
                            pp  = (uit**2 + vit**2)**0.5
                            img = ax.tricontourf(triang, pp, clev, cmap=cmap, norm=norm, extend=opt)
                            #-generate a quiver plot 
                            quiv = gen_quiv(uit,vit,pp,N,E,S,W)
                            #quiv.to_csv("out.csv",index=False)
                            #-set max quiver scale
                            if unit=='si': scale = 50
                            else:          scale = 100
                            #-add wave direction
                            plt.quiver(quiv['LON'].astype(float),quiv['LAT'].astype(float),
                                       quiv['u'].astype(float),quiv['v'].astype(float),color='darkgreen',
                                       zorder=500,scale=scale,scale_units="inches")
                            #-calculate dimensions of subplot
                            axbbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                            width, height = axbbox.width, axbbox.height
                            #-add a scale arrow and text
                            x_proj, y_proj = ax.projection.transform_point(W, N, crs.PlateCarree())
                            x_disp, y_disp = ax.transData.transform((x_proj, y_proj))
                            x_axes, y_axes = ax.transAxes.inverted().transform((x_disp,y_disp))
                            if unit=='si':
                                ax.arrow(x_axes+0.01,y_axes-0.02,1/width,0,head_width=0.012,length_includes_head=True,transform=ax.transAxes,zorder=501)
                                ax.text (x_axes+0.01,y_axes-0.04,"2.54 cm: "+str(scale)+" "+units,va='top',ha='left',transform=ax.transAxes,zorder=501)
                            else:
                                ax.arrow(x_axes+0.01,y_axes-0.02,1/width,0,head_width=0.012,length_includes_head=True,transform=ax.transAxes,zorder=501)
                                ax.text (x_axes+0.01,y_axes-0.04,"1 inch: "+str(scale)+" "+units,va='top',ha='left',transform=ax.transAxes,zorder=501)
                        else:
                            #-title label
                            title = "Barometric Pressure""\n"+"Valid:"+dates[d].strftime("%Y-%m-%dT%H:%M:%S UTC")
                            #-wave heights
                            if unit=='imperial':
                                #-units label
                                units = "mb"
                                #-color levels
                                clev    = list(np.arange(900,1050+2,2))
                                cntlevs = list(np.arange(900,1050+5,5))
                            else:
                                #-units label
                                units = "pa"
                                #-color levels
                                clev    = list(np.arange(900*100,(1050+2)*100,2*100))
                                cntlevs = list(np.arange(900*100,1050*100+5*100,5*100))
                            #-add label
                            label   = "barometric pressure ("+units+")"
                            #-color scheme
                            cmap    = mpl.cm.turbo
                            norm    = mpl.colors.BoundaryNorm(clev, cmap.N, extend='both')
                            #-set extend option for colorbar
                            opt   = 'max'
                            #-plot shaded image
                            img = ax.tricontourf(triang, pit, clev, cmap=cmap, norm=norm, extend=opt)   
                            if f==1:
                                #-add contour labels
                                cnt = ax.tricontour(triang, pit, cntlevs, colors='purple')
                                #-add labels
                                cla = ax.clabel(cnt,colors='purple',manual=False,inline=True,fmt=' {:.0f} '.format,use_clabeltext=True)
                                #-add halo around labels
                                plt.setp(cla, path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])
                        #-colorbar generator
                        def make_colorbar(ax, mappable, label):
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes('bottom','5%',pad='5%',axes_class=mpl.pyplot.Axes)
                            cbar = ax.get_figure().colorbar(mappable,cax=cax,orientation='horizontal',label=label,ticks=clev[::2])
                            cbar.ax.set_xticklabels(labels=clev[::2],rotation=-35)
                        #-plot title
                        ax.set_title(title,fontsize=16,fontweight='bold',loc='center')
                        #-plot max min points
                        if f==0: 
                            min_max(wsit,0,1)
                            #-plot max and min strings
                            ax.text(0.33,0.01,"Max: "+maxs,transform=ax.transAxes,ha='center',va='bottom', fontsize = 16, fontweight='bold',zorder=100000,c='r',
                                    path_effects=[pe.withStroke(linewidth=2, foreground="w")])
                        else: 
                            min_max(pit,1,0)
                            #-plot max and min strings
                            ax.text(0.66,0.01,"Min: "+mins,transform=ax.transAxes,ha='center'  ,va='bottom', fontsize = 16, fontweight='bold',zorder=100000,c='b',
                                    path_effects=[pe.withStroke(linewidth=2, foreground="w")])
                        #-add graticule
                        gl = ax.gridlines(crs=crs.PlateCarree(),draw_labels=True,linewidth=2,color='gray',alpha=0.5,linestyle='--')
                        #gl.xlabels_top   = False
                        if f==0:
                            gl.top_labels = False
                            gl.right_labels = False
                            #gl.ylabels_left  = True
                            #gl.ylabels_right = False
                        if f==1:
                            gl.top_labels = False
                            gl.left_labels = False
                            #gl.ylabels_left  = False
                            #gl.ylabels_right = True
                        gl.xformatter = LONGITUDE_FORMATTER
                        gl.yformatter = LATITUDE_FORMATTER
                        gl.ylabel_style = {'size': 12, 'color': 'blue','weight': 'bold'}
                        gl.xlabel_style = {'size': 12, 'color': 'red' ,'weight': 'bold'}
                        #-add colorbar
                        make_colorbar(ax, img, label)
                        if f==1:
                            #-adjust figure
                            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.99, wspace=0.05, hspace=0.1)
                            #-save figure
                            plt.savefig(out,dpi=300,facecolor='w',edgecolor='w')
                            plt.close(fig=None)
                #-water level and wave height figure from adcirc 
                if ((o==1) and (os.path.exists( (path + "/images/"+dates[d].strftime("BB-%Y%m%d%H%M%S.png")).replace("//","/").replace("\\","/") )==False)):
                    #-output image
                    out = "./images/"+dates[d].strftime("BB-%Y%m%d%H%M%S.png")
                    #-subset fields 
                    #-adcirc
                    zit = np.where(h2O ['zeta' ][d,:].data==-99999,0,h2O ['zeta' ][d,:].data)*zeta
                    uit = np.where(h2Od['u-vel'][d,:].data==-99999,0,h2Od['u-vel'][d,:].data)*wz
                    vit = np.where(h2Od['v-vel'][d,:].data==-99999,0,h2Od['v-vel'][d,:].data)*wz
                    #-swan
                    wit = np.where(wvht['swan_HS'][d,:].data==-99999,0,wvht['swan_HS'][d,:].data)*wvc 
                    dit = np.where(wvd['swan_DIR'][d,:].data==-99999,0,wvd['swan_DIR'][d,:].data)
                    #-inform
                    print("Generating Figure: "+out)
                    #-create figure
                    fig = plt.figure(figsize=(16, 9))
                    for f in range(2):
                        #-determine min and max of the file
                        if f==0:
                            if vortex==1:
                                #-subset
                                log       = points.intersects(geom) & (h2O['zeta' ][d,:].data!=-99999)
                                maxs,mins = str(np.round(np.nanmax(zit[log]),1)),str(np.round(np.nanmin(zit[log]),1))
                                #-determine row of min and max
                                maxr = np.nanargmax(np.where(log==True,zit,np.nan))
                                minr = np.nanargmin(np.where(log==True,zit,np.nan))
                            else:
                                #-subset
                                log       = h2O['zeta' ][d,:].data!=-99999
                                maxs,mins = str(np.round(np.nanmax(zit[log]),1)),str(np.round(np.nanmin(zit[log]),1))
                                #-determine row of min and max
                                maxr = np.nanargmax(np.where(log==True,zit,np.nan))
                                minr = np.nanargmin(np.where(log==True,zit,np.nan))
                        else:
                            if vortex==1:
                                #-subset
                                log       = points.intersects(geom) & (wvht['swan_HS'][d,:].data!=-99999)
                                maxs,_ = str(np.round(np.nanmax(wit[log]),1)),str(np.round(np.nanmin(wit[log]),1))
                                #-determine row of min and max
                                maxr =  np.nanargmax(np.where(log==True,wit,np.nan))
                            else:
                                #-subset
                                log       = wvht['swan_HS'][d,:].data!=-99999
                                maxs,_ = str(np.round(np.nanmax(wit[log]),1)),str(np.round(np.nanmin(wit[log]),1))
                                #-determine row of min and max
                                maxr = np.nanargmax(np.where(log==True,wit,np.nan))
                        #-create axis
                        ax = fig.add_subplot(1, 2, f+1, projection=crs.PlateCarree())
                        ax.set_extent([W,E,S,N],crs=crs.PlateCarree())
                        #-add mapping resources
                        ax.add_feature(coast,edgecolor='black',lw=2.0)
                        #-create triangulation
                        if ((bbox!=None) or (vortex==1)):
                            #-masking by geom
                            triang = tri.Triangulation(lon1d,lat1d,triangles=triangles,mask=~geolog)
                        else:
                            #-no masking
                            triang = tri.Triangulation(lon1d,lat1d,triangles=triangles)
                        #-determine which field to plot 
                        if f==0:
                            #-define title
                            title = "Water Surface Elevation and Flow Rate""\n"+"Valid: "+dates[d].strftime("%Y-%m-%dT%H:%M:%S UTC")
                            #-water levels
                            if unit=='imperial':
                                #-units label
                                units = "feet"
                                #-color levels
                                clev    = list(np.arange(-10,10.25,0.25))
                                cntlevs = list(np.arange(1  ,10+1,1))
                            else:
                                #-units label
                                units = "meters"
                                #-color levels
                                clev    = list(np.arange(-5,5.1,0.1))
                                cntlevs = list(np.arange(0.5,3+0.5,0.5))
                            #-define label
                            label   = "water surface elevation ("+units+")"
                            #-generate colorbar
                            cmap    = mpl.cm.seismic
                            norm    = mpl.colors.BoundaryNorm(clev, cmap.N, extend='both')
                            #-set extend option for colorbar
                            opt     = 'both'
                            #-plot shaded image
                            img = ax.tricontourf(triang, zit, clev, cmap=cmap, norm=norm, extend=opt)
                            #-add contour labels
                            cnt = ax.tricontour(triang, zit, cntlevs, colors='purple')
                            #-add labels
                            cla = ax.clabel(cnt,colors='purple',manual=False,inline=True,fmt=' {:.0f} '.format,use_clabeltext=True)
                            #-add halo around labels
                            plt.setp(cla, path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])
                            #-generate a quiver plot
                            flow = ( uit**2 + vit**2 )**0.5 
                            quiv = gen_quiv(uit,vit,flow,N,E,S,W)
                            #-set max quiver scale
                            if unit=='si': scale = 0.75
                            else:          scale = 1.5
                        else:
                            #-title label
                            title = "Wave Height and Wave Direction""\n"+"Valid: "+dates[d].strftime("%Y-%m-%dT%H:%M:%S UTC")
                            #-wave heights
                            if unit=='imperial':
                                #-units label
                                units = "feet"
                                #-color levels
                                clev    = [0,6,12,18,24,30,36,42]
                                cntlevs = [6,12,18,24,30,36,42]
                            else:
                                #-units label
                                units = "meters"
                                #-color levels
                                clev    = [0,2,4,6,8,10,12,14]
                                cntlevs = [2,4,6,8,10,12,14]
                            #-add label
                            label   = "wave height ("+units+")"
                            #-color scheme
                            colors  = ["white","gray","darkblue","yellow","orange","red","magenta","pink"]
                            norm    = plt.Normalize(min(clev) ,max(clev))
                            tuples  = list(zip(map(norm,clev) , colors  ))
                            cmap    = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
                            #-fix clev array for finer tuning of colorscheme
                            if unit=='si': clev = np.arange(0,clev[-1]+0.25,0.25)
                            else:          clev = np.arange(0,clev[-1]+1   ,   1)
                            #-set extend option for colorbar
                            opt   = 'max'
                            #-plot shaded image
                            img = ax.tricontourf(triang, wit, clev, cmap=cmap, norm=norm, extend=opt)    
                            #-add contour labels
                            cnt = ax.tricontour(triang, wit, cntlevs, colors='purple')
                            #-add labels
                            cla = ax.clabel(cnt,colors='purple',manual=False,inline=True,fmt=' {:.0f} '.format,use_clabeltext=True)
                            #-add halo around labels
                            plt.setp(cla, path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])
                            #-calculate components 
                            uit,vit = get_uv(wit,dit)
                            #-generate a quiver plot 
                            quiv = gen_quiv(uit,vit,wit,N,E,S,W)
                            #-set max quiver scale
                            if unit=='si': scale = 10 
                            else:          scale = 40 
                        #-colorbar generator
                        def make_colorbar(ax, mappable, label):
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes('bottom','5%',pad='5%',axes_class=mpl.pyplot.Axes)
                            cbar = ax.get_figure().colorbar(mappable,cax=cax,orientation='horizontal',label=label,ticks=clev[::2])
                            cbar.ax.set_xticklabels(labels=clev[::2],rotation=-35)
                        #-add wave direction
                        plt.quiver(quiv['LON'].astype(float),quiv['LAT'].astype(float),
                                   quiv['u'].astype(float),quiv['v'].astype(float),color='darkgreen',
                                   zorder=500,scale=scale,scale_units="inches")
                        #-calculate dimensions of subplot
                        axbbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                        width, height = axbbox.width, axbbox.height
                        #-add a scale arrow and text
                        x_proj, y_proj = ax.projection.transform_point(W, N, crs.PlateCarree())
                        x_disp, y_disp = ax.transData.transform((x_proj, y_proj))
                        x_axes, y_axes = ax.transAxes.inverted().transform((x_disp,y_disp))
                        if f==0:
                            if unit=='si':
                                ax.arrow(x_axes+0.01,y_axes-0.02,1/width,0,head_width=0.012,length_includes_head=True,transform=ax.transAxes,zorder=501)
                                ax.text (x_axes+0.01,y_axes-0.04,"2.54 cm: "+str(scale)+" m/s",va='top',ha='left',transform=ax.transAxes,zorder=501)
                            else:
                                ax.arrow(x_axes+0.01,y_axes-0.02,1/width,0,head_width=0.012,length_includes_head=True,transform=ax.transAxes,zorder=501)
                                ax.text (x_axes+0.01,y_axes-0.04,"1 inch: "+str(scale)+" miles/hour",va='top',ha='left',transform=ax.transAxes,zorder=501)
                        else:
                            if unit=='si':
                                ax.arrow(x_axes+0.01,y_axes-0.02,1/width,0,head_width=0.012,length_includes_head=True,transform=ax.transAxes,zorder=501)
                                ax.text (x_axes+0.01,y_axes-0.04,"2.54 cm: "+str(scale)+" "+units,va='top',ha='left',transform=ax.transAxes,zorder=501)
                            else:
                                ax.arrow(x_axes+0.01,y_axes-0.02,1/width,0,head_width=0.012,length_includes_head=True,transform=ax.transAxes,zorder=501)
                                ax.text (x_axes+0.01,y_axes-0.04,"1 inch: "+str(scale)+" "+units,va='top',ha='left',transform=ax.transAxes,zorder=501)
                        #-plot title
                        ax.set_title(title,fontsize=16,fontweight='bold',loc='center')
                        #-plot max min points
                        if f==0: 
                            min_max(zit,1,1)
                            #-plot max and min strings
                            ax.text(0.33,0.01,"Max: "+maxs,transform=ax.transAxes,ha='center',va='bottom', fontsize = 16, fontweight='bold',zorder=100000,c='r',
                                    path_effects=[pe.withStroke(linewidth=2, foreground="w")])
                            ax.text(0.66,0.01,"Min: "+mins,transform=ax.transAxes,ha='center'  ,va='bottom', fontsize = 16, fontweight='bold',zorder=100000,c='b',
                                    path_effects=[pe.withStroke(linewidth=2, foreground="w")])
                        else: 
                            min_max(wit,0,1)
                            #-plot max and min strings
                            ax.text(0.33,0.01,"Max: "+maxs,transform=ax.transAxes,ha='center',va='bottom', fontsize = 16, fontweight='bold',zorder=100000,c='r',
                                    path_effects=[pe.withStroke(linewidth=2, foreground="w")])
                        #-add graticule
                        gl = ax.gridlines(crs=crs.PlateCarree(),draw_labels=True,linewidth=2,color='gray',alpha=0.5,linestyle='--')
                        #gl.xlabels_top   = False
                        if f==0:
                            gl.top_labels = False
                            gl.right_labels = False
                            #gl.ylabels_left  = True
                            #gl.ylabels_right = False
                        if f==1:
                            gl.top_labels = False
                            gl.left_labels = False
                            #gl.ylabels_left  = False
                            #gl.ylabels_right = True
                        gl.xformatter = LONGITUDE_FORMATTER
                        gl.yformatter = LATITUDE_FORMATTER
                        gl.ylabel_style = {'size': 12, 'color': 'blue','weight': 'bold'}
                        gl.xlabel_style = {'size': 12, 'color': 'red' ,'weight': 'bold'}
                        #-add colorbar
                        make_colorbar(ax, img, label)
                        if f==1:
                            #-adjust figure
                            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.99, wspace=0.05, hspace=0.1)
                            #-save figure
                            plt.savefig(out,dpi=300,facecolor='w',edgecolor='w')
                            plt.close(fig=None)

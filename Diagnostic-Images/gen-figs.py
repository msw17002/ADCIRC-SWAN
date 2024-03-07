#!/home/msw17002/anaconda3/bin/python

import warnings
warnings.filterwarnings("ignore")

#===this function calculates u/v components from direction and intensity
def wind_spddir_to_uv(mag,dir):
    """
    calculated the u and v wind components from wind speed and direction
    Input:
        wspd: wind speed
        wdir: wind direction
    Output:
        u: u wind component
        v: v wind component
    """
    rad = 4.0*np.arctan(1)/180.
    u = -mag*np.sin(rad*dir)
    v = -mag*np.cos(rad*dir)
    return u,v
#===this function plots the max. and min. value onto a projected map 
def min_max(zfield,minopt,maxopt):
    #-determine row of min and max
    minr = np.nanargmin(zfield)
    maxr = np.nanargmax(zfield)
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
#===this function plots the max. and min. numerics on the plot axis
def plot_maxmin(lonlat,zfield,minopt):
    #-clean field
    zfield              = zfield[((lonlat[:,0]>W) & (lonlat[:,0]<E) & (lonlat[:,1]<N) & (lonlat[:,1]>S))]
    zfield[zfield<-100] = np.nan
    if all(np.isnan(zfield)):
        ax.text(0.80,0.035,"Max",fontsize=13,transform=ax.transAxes,ha='left',va='bottom',
                fontweight='bold',color='r',path_effects=[pe.withStroke(linewidth=1, foreground='k')])
        ax.text(0.865,0.035,": N/A",fontsize=13,transform=ax.transAxes,ha='left',va='bottom',
                fontweight='bold',color='r',path_effects=[pe.withStroke(linewidth=1, foreground='k')])
        if minopt==1:
            ax.text(0.80,0.01,"Min",fontsize=13,transform=ax.transAxes,ha='left',va='bottom',
                    fontweight='bold',color='b',path_effects=[pe.withStroke(linewidth=1, foreground='k')])
            ax.text(0.865,0.01,": N/A",fontsize=13,transform=ax.transAxes,ha='left',va='bottom',
                    fontweight='bold',color='b',path_effects=[pe.withStroke(linewidth=1, foreground='k')])
    else:
        lonlat = lonlat[((lonlat[:,0]>W) & (lonlat[:,0]<E) & (lonlat[:,1]<N) & (lonlat[:,1]>S))]
        idx    = np.nanargmin(zfield)
        zmin   = zfield[idx]
        idx    = np.nanargmax(zfield)
        zmax   = zfield[idx]
        ax.text(0.80,0.035,"Max",fontsize=13,transform=ax.transAxes,ha='left',va='bottom',
                fontweight='bold',color='r',path_effects=[pe.withStroke(linewidth=1, foreground='k')])
        ax.text(0.865,0.035,": "+str(np.round(zmax,1)),fontsize=13,transform=ax.transAxes,ha='left',va='bottom',
                fontweight='bold',color='r',path_effects=[pe.withStroke(linewidth=1, foreground='k')])
        if minopt==1:
            ax.text(0.80,0.01,"Min",fontsize=13,transform=ax.transAxes,ha='left',va='bottom',
                              fontweight='bold',color='b',path_effects=[pe.withStroke(linewidth=1, foreground='k')])
            ax.text(0.865,0.01,": "+str(np.round(zmin,1)),fontsize=13,transform=ax.transAxes,ha='left',va='bottom',
                              fontweight='bold',color='b',path_effects=[pe.withStroke(linewidth=1, foreground='k')])
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

#===load libraries
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.interpolate as interpolate
from shapely.geometry import Polygon, Point
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import cartopy.io.img_tiles as cimgt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from loguru import logger
import cartopy.crs as crs
#import matplotlib.colors
import matplotlib as mpl
import pandas as pd 
import numpy as np
import geopandas
import argparse
import datetime 
import netCDF4
import sys
import os

import matplotlib.tri as tri
import cartopy.feature as cfeature
import shapely.geometry
import matplotlib
import pyproj

bbox = str("43:-67:37.5:-78.4")

#---create arguments
parser = argparse.ArgumentParser()
parser.add_argument('-path', '--path', nargs ="+", required=True, help='Path to where maxele.63.nc",maxwvel.63.nc",minpr.63.nc, and or swan_HS_max.63.nc are located')
parser.add_argument('-unit', '--unit', nargs ="+", required=True, help='Either imperial (feet) or si (meters)')
#-obtain arguments
args = parser.parse_args()
path = str(args.path[0])
unit = str(args.unit[0])

if ((unit=='imperial') or (unit=='si')): pass
else:
    logger.info("Script terminating!")
    sys.exit("'-unit' option is either 'imperial' (feet) or 'si' (meters)")


#---read global shapefil
coast = cfeature.GSHHSFeature(scale="f")
#---query fort.14 file path
indir  = "./"
#---initiate grid sequence 
points = pd.DataFrame([])
#---plot through all temporal independant, diagnostic fields
diags  = ["maxele.63.nc","maxwvel.63.nc","minpr.63.nc","swan_HS_max.63.nc"]
diags  = ["swan_HS_max.63.nc"]
diags  = ["maxwvel.63.nc"]
#---main script
for d in diags:
    if os.path.exists(path+"/"+d):
        logger.info("Generating: "+path+d.replace(".nc",".png"))
        #---read netcdf4 file 
        nc = netCDF4.Dataset(path+"/"+d)
        #---generate grid if it hasn't been created
        if points.shape==(0,0):
            logger.info("Building Mesh & Node Structure from: "+path+d+"!")
            #-grid coordinates
            lon1d   = np.asarray(nc['x'][:])
            lat1d   = np.asarray(nc['y'][:])
            #-triangular mesh
            npgrid = np.asarray(nc['element'][:,:])
            lonlat = np.transpose(np.vstack((lon1d,lat1d)))
            #-get triangle vorticies. create geopandas dataframe
            # of triangular mesh
            triangles = []
            if bbox!="": mesh = []
            for node in npgrid:
                if bbox!="":
                    #-determine indicies
                    i0   = lonlat[int(node[0]-1)]
                    i1   = lonlat[int(node[1]-1)]
                    i2   = lonlat[int(node[2]-1)]
                    #-create a polygon
                    mesh.append(Polygon(zip([i0[0],i1[0],i2[0],i0[0]], [i0[1],i1[1],i2[1],i0[1]])))
                    #-for contouring
                    triangles.append([int(node[0]-1),int(node[1]-1),int(node[2]-1)])
            if bbox!="":
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
            if bbox=="":
                #-generate bounds 
                N,E,S,W = np.max(points['LAT']),np.max(points['LON']),np.min(points['LAT']),np.min(points['LON'])
                #-define extent label
                labelex = "Full Domain Extent"
            else: 
                #-generate bounds
                N,E,S,W = bbox.split(":")
                N,E,S,W = float(N),float(E),float(S),float(W)
                #-define extent label 
                labelex = "Domain Subset"
            #-create points to calculate distances
            PNW = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[Point( W,N )]).to_crs('EPSG:3857')
            PSW = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[Point( W,S )]).to_crs('EPSG:3857')
            PSE = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[Point( E,S )]).to_crs('EPSG:3857')
            #-determine horizontal distance
            dx = PNW.distance(PSE)[0]
            #-determine the latitudinal distance
            dy = PSE.distance(PSW)[0]
            #generate polygon of bounds
            geom   = Polygon(zip([W,E,E,W,W], [N,N,S,S,N]))
            pol    = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geom])
            #-get times
            try:
                dtostr = datetime.datetime.strptime(nc['time'].units.split("!")[0].strip(),"seconds since %Y-%m-%d %H:%M:%S")
                dtestr = dtostr + datetime.timedelta(hours=float(nc['time'][0])/(60*60))
            except:
                dtostr = ""
                dtestr = ""
            #-create triangulation
            if bbox!="":             
                #-no masking
                triang = tri.Triangulation(lon1d,lat1d,triangles=triangles)
            else: 
                #-masking by geom
                triang = tri.Triangulation(lon1d,lat1d,triangles=triangles,mask=~mesh.intersects(geom))
        #---create mapping resources
        if d=="maxele.63.nc":
            #-define title
            title = "ADCIRC: Maximum Water Surface Elevation (feet)"
            #-define label 
            label   = "maximum water surface elevation (feet)"
            #-get and convert to array
            zfield  = nc['zeta_max'][:].data
            #zfield  = np.where(zfield==-99999.0,np.nan,zfield)
            zfield  = zfield*3.28084
            #-generate colorbar
            clev    = list(np.arange(-10,10.25,0.25))
            cmap    = mpl.cm.seismic
            norm    = mpl.colors.BoundaryNorm(clev, cmap.N, extend='both')
            #-set extend option for colorbar
            opt     = 'both'

        if d=="maxwvel.63.nc":
            #-title label
            title = "Maximum Water Surace Wind Velocity"
            #-get and convert to array
            zfield  = nc['wind_max'][:].data
            #-get and convert to array
            if   ((nc['wind_max'].units=='m s-1') and (unit=='si')):        cnv = 1         #meter to meter
            elif ((nc['wind_max'].units=='m s-1') and (unit=='imperial')):  cnv = 2.23694   #meter to foot
            elif ((nc['wind_max'].units!='m s-1') and (unit=='imperial')):  cnv = 1         #foot to foot
            elif ((nc['wind_max'].units!='m s-1') and (unit=='si')):        cnv = 1/2.23694 #foot to meter
            else: sys.exit("Conversion b/n "+nc[d].units+" to "+unit+" isn't cataloged in script! Must append conversion!!!")
            #-max/min dataframes
            if bbox!="":
                #-full domain
                maxs,mins = str(np.round(np.nanmax(np.where(zfield==-99999,np.nan,zfield))*cnv,1)), \
                            str(np.round(np.nanmin(np.where(zfield==-99999,np.nan,zfield))*cnv,1))
            else:
                #-subset
                log       = ((points.intersects(geom)) & (zfield!=-99999))
                maxs,mins = str(np.round(np.nanmax(zfield[log])*cnv,1)),str(np.round(np.nanmin(zfield[log])*cnv,1))
            #-convert
            zfield  = zfield*cnv
            #-create triangulation
            if bbox!="":
                #-no masking
                triang = tri.Triangulation(lon1d,lat1d,triangles=triangles)
            else:
                #-masking by geom
                triang = tri.Triangulation(lon1d,lat1d,triangles=triangles,mask=~log)
            #-conditional labeling
            if unit=='imperial':
                #-units label
                units = "miles/hour"
                #-color levels
                clev  = np.arange(0,100+2,2)
            else:
                #-units label
                units = "meters/second"
                #-color levels
                clev  = np.arange(0,50+1,1)
            #-set the label 
            label   = "wind speed ("+units+")"
            #-generate colorbar
            colors1 = plt.cm.Greys(np.linspace(0., 0.5, 5))   #5,10,15,20,25,30,35
            colors2 = plt.cm.winter(np.linspace(0.5, 1, 10)) #Blues plt.cm.Greens_r(np.linspace(0, 1, 128))
            colors3 = plt.cm.YlOrRd(np.linspace(0., 1, 10))
            colors4 = plt.cm.gnuplot2(np.linspace(0.,0.9, 15))
            colors  = np.vstack((colors1, colors2, colors3, colors4))
            cmap    = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
            clev    = np.arange(0,100+2,2)
            norm    = mpl.colors.BoundaryNorm(clev, cmap.N, extend='max')
            #-set extend option for colorbar
            opt     = 'max'

        if d=="minpr.63.nc":
            #-title label
            title   = "Minimum Barometric Pressure"
            #-colorbar label
            label   = "minimum barometric pressure (mb)" 
            units   = "mb"
            #-get and convert to array
            zfield  = nc['pressure_min'][:].data
            #-conversion factor should always be constant here
            cnv     = 98.04139432
            #-max/min dataframes
            if bbox!="":
                #-full domain
                maxs,mins = str(np.round(np.nanmax(np.where(zfield==-99999,np.nan,zfield))*cnv,0)), \
                            str(np.round(np.nanmin(np.where(zfield==-99999,np.nan,zfield))*cnv,0))
            else: 
                #-subset
                log       = ((points.intersects(geom)) & (zfield!=-99999))
                maxs,mins = str(np.round(np.nanmax(zfield[log])*cnv,0)),str(np.round(np.nanmin(zfield[log])*cnv,0))
            #-convert 
            zfield  = zfield*cnv
            #-create triangulation
            if bbox!="":
                #-no masking
                triang = tri.Triangulation(lon1d,lat1d,triangles=triangles)
            else:
                #-masking by geom
                triang = tri.Triangulation(lon1d,lat1d,triangles=triangles,mask=~log)
            #-generate colorbar
            clev    = list(np.arange(900,1050+2,2))
            cmap    = mpl.cm.turbo
            norm    = mpl.colors.BoundaryNorm(clev, cmap.N, extend='both')
            #-set extend option for colorbar
            opt     = 'both'

        if d=="swan_HS_max.63.nc":
            #-title label
            title = "Maximum Significant Wave Height and Wave Direction"
            #-get and convert to array
            if   ((nc['swan_HS_max'].units=='m') and (unit=='si')):        cnv = 1         #meter to meter 
            elif ((nc['swan_HS_max'].units=='m') and (unit=='imperial')):  cnv = 3.28084   #meter to foot
            elif ((nc['swan_HS_max'].units!='m') and (unit=='imperial')):  cnv = 1         #foot to foot
            elif ((nc['swan_HS_max'].units!='m') and (unit=='si')):        cnv = 1/3.28084 #foot to meter
            else: sys.exit("Conversion b/n "+nc['swan_HS_max'].units+" to "+unit+" isn't cataloged in script! Must append conversion!!!")
            #-get and convert array
            zfield  = nc['swan_HS_max'][:].data
            #-max/min dataframes
            if bbox!="":
                #-full domain
                maxs,mins = str(np.round(np.nanmax(np.where(zfield==-99999,np.nan,zfield))*cnv,1)), \
                            str(np.round(np.nanmin(np.where(zfield==-99999,np.nan,zfield))*cnv,1))
            else:
                #-subset
                log       = ((points.intersects(geom)) & (zfield!=-99999))
                maxs,mins = str(np.round(np.nanmax(zfield[log])*cnv,1)),str(np.round(np.nanmin(zfield[log])*cnv,1))
            #-convert
            zfield  = zfield*cnv
            #-create triangulation
            if bbox!="":
                #-no masking
                triang = tri.Triangulation(lon1d,lat1d,triangles=triangles)
            else:
                #-masking by geom
                triang = tri.Triangulation(lon1d,lat1d,triangles=triangles,mask=~log)
            #-conditional labeling
            if unit=='imperial':
                #-units label
                units = "feet"
                #-color levels
                clev  = [0,6,12,18,24,30,36,42]
            else:
                #-units label
                units = "meters"
                #-color levels
                clev  = [0,2,4,6,8,10,12,14]
            #-add label
            label   = "maximum significant wave height ("+units+")"
            #-color scheme
            colors  = ["lightgray","lightblue","darkblue","yellow","orange","red","magenta","pink"]
            norm    = plt.Normalize(min(clev) ,max(clev))
            tuples  = list(zip(map(norm,clev) , colors  ))
            cmap    = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
            #-fix clev array for finer tuning of colorscheme
            if unit=='si': clev = np.arange(0,clev[-1]+0.25,0.25)
            else:          clev = np.arange(0,clev[-1]+1   ,   1)
            #-also read max. flow direciton 
            ncdir  = netCDF4.Dataset(path+"/"+"swan_DIR_max.63.nc")
            #-get field
            dirmax = np.asarray(ncdir['swan_DIR_max'][:])
            #-close file
            ncdir.close()
            #-create u/v components
            waveU,waveV = [],[]
            for i in range(len(dirmax)):
                #-calculate wave compenents 
                if zfield[i]==-99999: u,v = 0,0
                else:                 u,v = wind_spddir_to_uv(zfield[i],dirmax[i])
                #-append to lists 
                waveU.append(u)
                waveV.append(v)
            #-convert to arrays
            waveU = np.asarray(waveU)
            waveV = np.asarray(waveV)
            #-generate quiver
            quiv  = gen_quiv(waveU,waveV,zfield,N,E,S,W)            
            #-set extend option for colorbar
            opt   = 'max'

        #===================================================================================================
        #---Create a figure
        #-figure
        fig  = plt.figure(figsize=(10,9))
        #===maximum water level===================================================================================================
        #-map projection + extent
        ax   = fig.add_subplot(1,1,1,projection=crs.PlateCarree())
        ax.set_extent([W-0.2, E+0.2, S-0.1, N+0.1])
        #-plot title 
        ax.set_title(title,fontsize=16,fontweight='bold',loc='center')
        #----------------------------------------------------------------------------------------------------------------------
        def make_colorbar(ax, mappable, label):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right','5%',pad='2%',axes_class=mpl.pyplot.Axes)
            cbar = ax.get_figure().colorbar(mappable,cax=cax,orientation='vertical',label=label,
                                            ticks=clev[::2])
            #cbar.ax.set_xticklabels(labels=clev[::2],rotation=-25)
        #-add mapping resources
        ax.add_feature(coast,edgecolor='black',lw=2.0)
        #-plot shaded image
        img = ax.tricontourf(triang, zfield, clev, cmap=cmap, norm=norm, extend=opt)
        if d=='maxele.63.nc'     : cntlevs = list(np.arange(1,10+1,1))
        if d=='minpr.63.nc'      : cntlevs = list(np.arange(900,1050+5,5))
        if d=='swan_HS_max.63.nc': 
            if unit=='si': cntlevs = [2,4,6,8,10,12,14]
            else:          cntlevs = [6,12,18,24,30,36,42]
        if (d=='maxele.63.nc' or d=='minpr.63.nc' or d=='swan_HS_max.63.nc'): 
            #-add contour labels
            cnt = ax.tricontour(triang, zfield, cntlevs, colors='purple')
            #-add labels
            cla = ax.clabel(cnt,colors='purple',manual=False,inline=True,fmt=' {:.0f} '.format,use_clabeltext=True)
            #-add halo around labels
            plt.setp(cla, path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])
        if d=='swan_HS_max.63.nc': 
            #-set max quiver scale to 2 inches
            scale = int(10*round( np.nanmax(quiv['z']) / 10 ))
            #-add wave direction
            plt.quiver(quiv['LON'].astype(float),quiv['LAT'].astype(float), 
                       quiv['u'].astype(float),quiv['v'].astype(float),color='darkgreen',
                       zorder=500,scale=scale,scale_units="inches")
            #-calculate dimensions of subplot
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width, height = bbox.width, bbox.height
            #-add a scale arrow and text
            x_proj, y_proj = ax.projection.transform_point(W, N, crs.PlateCarree())
            x_disp, y_disp = ax.transData.transform((x_proj, y_proj))
            x_axes, y_axes = ax.transAxes.inverted().transform((x_disp,y_disp))
            ax.arrow(x_axes+0.01,y_axes-0.04,1/width,0,head_width=0.012,length_includes_head=True,transform=ax.transAxes,zorder=501)
            ax.text (x_axes+0.01,y_axes-0.06,"1 inch: "+str(scale)+" "+units,va='top',ha='left',transform=ax.transAxes,zorder=501)

        #-add analysis domain
        ax.add_geometries(pol['geometry'], crs.PlateCarree(), facecolor="none", edgecolor='red', lw=2.0,label='analysis region')
        #-add graticule
        gl = ax.gridlines(crs=crs.PlateCarree(),draw_labels=True,linewidth=2,color='gray',alpha=0.5,linestyle='--')
        gl.xlabels_top   = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylabel_style = {'size': 12, 'color': 'blue','weight': 'bold'}
        gl.xlabel_style = {'size': 12, 'color': 'red' ,'weight': 'bold'}
        #-reset axes values 
        x_proj, y_proj = ax.projection.transform_point(W, S, crs.PlateCarree())
        x_disp, y_disp = ax.transData.transform((x_proj, y_proj))
        x_axes, y_axes = ax.transAxes.inverted().transform((x_disp,y_disp))
        #-max and min strings
        if d=="maxele.63.nc": 
            #-rounding
            maxs,mins = str(np.round(np.nanmax(zfield),1)),str(np.round(np.nanmin(zfield),1))
            #-plot points 
            min_max(zfield,1,1)
            #-plot max and min strings
            ax.text(0.01,y_axes+0.01,"Min: "+mins,transform=ax.transAxes,ha='left'  ,va='bottom', fontsize = 12, fontweight='bold',zorder=100000,c='b',
                    path_effects=[pe.withStroke(linewidth=2, foreground="w")])
            ax.text(0.50,y_axes+0.01,"Max: "+maxs,transform=ax.transAxes,ha='center',va='bottom', fontsize = 12, fontweight='bold',zorder=100000,c='r',
                    path_effects=[pe.withStroke(linewidth=2, foreground="w")])
        else:  
            #-plot points 
            if (d=='maxwvel.63.nc' or d=='minpr.63.nc'): min_max(zfield,1,1)
            else:                                        min_max(zfield,0,1)
            #-plot max and min strings
            ax.text(0.50,y_axes+0.01,"Max: "+maxs,transform=ax.transAxes,ha='center',va='bottom', fontsize = 12, fontweight='bold',zorder=100000,c='r',
                    path_effects=[pe.withStroke(linewidth=2, foreground="w")])
            if mins.replace(".0","")=="0": pass
            else:  
                ax.text(0.01,y_axes+0.01,"Min: "+mins,transform=ax.transAxes,ha='left',va='bottom', fontsize = 12, fontweight='bold',zorder=100000,c='b',
                        path_effects=[pe.withStroke(linewidth=2, foreground="w")])
        #-add a makeshift legend
        ax.text(0.99,y_axes+0.01,labelex,transform=ax.transAxes,ha='right',va='bottom', fontsize = 12, fontweight='bold',zorder=100000,c='r',
                path_effects=[pe.withStroke(linewidth=2, foreground="w")])
        #-add dumby point for colorbar then add colorbar
        make_colorbar(ax, img, label)
        #-adjust figure
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0., hspace=0.)
        #-save figure
        plt.savefig(d.replace(".nc","-")+unit+"-"+labelex.replace(" ","").lower()+".png",dpi=300,facecolor='w',edgecolor='w',bbox_inches='tight')
        plt.close(fig=None)
        #-update
        logger.info("Figure Generated: "+path+d.replace(".nc","-")+unit+"-"+labelex.replace(" ","").lower()+".png!")

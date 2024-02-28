<b>This script contains the following:</b> 

<h1>Configuring & Compiling of Dependencies and ADCRIC/SWAN</h1>
<p>1) comp-dependencies.sh  > compiles dependencies</p><br>
<p>2) run-config.sh         > compiles ADCIRC+SWAN </p><br>
<br>
To run,
<br>
a) copy both shell-scripts.
b) chmod +x file_name.sh
c) ./comp-dependencies.sh
d) ./run-config.sh

<h1>Generation of Images from Diagnostic Output from ADCIRC/SWAN</h1>
~/Diagnostic-Images/gen-figs.py
1) the following files are created from ADCIRC and or SWAN: maxele.63.nc, maxwvel.63.nc, minpr.63.nc, swan_HS_max.63.nc (must be accompanies by ).
2) maxele.63.nc (field 'zeta_max') units are meters (converts to feet)
3) maxwvel.63.nc (field 'wind_max') units are meters/second (convert to miles/hour)
4) minpr.63.nc (field 'pressure_min') units are meters of head (convert to milibar) 
5) swan_HS_max.63.nc (field 'swan_HS_max') units are meters (converts to feet)

To run,
python gen-figs.py -path path_to_files

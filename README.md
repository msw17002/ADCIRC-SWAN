<b>This script contains the following:</b> 

<h1>Configuring & Compiling of Dependencies and ADCRIC/SWAN</h1>
<ul>
    <li>comp-dependencies.sh  > compiles dependencies</li>
    <li>run-config.sh         > compiles ADCIRC+SWAN </li>
</ul>

<p>To run,</p>
<ul>
    <li>copy both shell-scripts.</li>
    <li>chmod +x file_name.sh</li>
    <li>./comp-dependencies.sh</li>
    <li>./run-config.sh</li>
</ul>
    
<h1>Generation of Images from Diagnostic Output from ADCIRC/SWAN</h1>
<p>~/Diagnostic-Images/gen-figs.py</p>
<ul>
    <li>the following files are created from ADCIRC and or SWAN: maxele.63.nc, maxwvel.63.nc, minpr.63.nc, swan_HS_max.63.nc (must be accompanies by swan_DIR_max.63.nc).</li>
    <li>maxele.63.nc (field 'zeta_max') units are meters (converts to feet)</li>
    <li>maxwvel.63.nc (field 'wind_max') units are meters/second (convert to miles/hour)</li>
    <li>minpr.63.nc (field 'pressure_min') units are meters of head (convert to milibar)</li>
    <li>swan_HS_max.63.nc (field 'swan_HS_max') units are meters (converts to feet)</li>
</ul>


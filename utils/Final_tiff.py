import os
from rasterio.crs import CRS

my_wkt="""PROJCRS["NAD_1983_Canada_Lambert",BASEGEOGCRS["NAD83",DATUM["North American Datum 1983", ELLIPSOID["GRS 1980",6378137,298.257222101004, LENGTHUNIT["metre",1]],ID["EPSG",6269]],PRIMEM["Greenwich",0,ANGLEUNIT["Degree",0.0174532925199433]]],CONVERSION["unnamed", METHOD["Lambert Conic Conformal (2SP)", ID["EPSG",9802]],PARAMETER["Latitude of false origin",0,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8821]], PARAMETER["Longitude of false origin",-95,ANGLEUNIT["Degree",0.0174532925199433], ID["EPSG",8822]], PARAMETER["Latitude of 1st standard parallel",49, ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude of 2nd standard parallel",77,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8824]],PARAMETER["Easting at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8826]],PARAMETER["Northing at false origin",0,LENGTHUNIT["metre",1], ID["EPSG",8827]]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1], LENGTHUNIT["metre",1,ID["EPSG",9001]]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]"""
# Create a CRS object from the WKT string
dst_projection = CRS.from_wkt(my_wkt)

def makemap(idresult, name, output):
    """
    Create and save raster maps for specified bands from the input DataFrame.

    Args:
        idresult (pd.DataFrame): DataFrame containing the data to map.
        name (str): Name to use in the output file names.
        output (str): Directory path where the output rasters will be saved.
    """
    # Select relevant columns from the DataFrame
    frametocompare = idresult[['x', 'y', 'Nancount', 'type']]
    bands = ['type', 'Nancount']

    for b in bands:
        print('Saving result for band:', b)

        # Prepare the DataFrame for the current band
        new_df = frametocompare[['y', 'x', b]].copy()
        new_df[b] = new_df[b].astype(float)

        # Convert DataFrame to xarray with spatial coordinates
        da = new_df.set_index(['y', 'x']).to_xarray()
        da = da.set_coords(['y', 'x'])
        da = da.rio.write_crs(dst_projection, inplace=True)

        # Define output paths
        tempoutRaster = os.path.join(output, f'temps_{name}_{b}.tif')
        outRaster = os.path.join(output, f'result_{name}_{b}.tif')

        # Save the temporary raster
        da.astype('int16').rio.to_raster(tempoutRaster, compress='LZW')

        # Use GDAL to warp and save the final raster
        warp_command = f'gdalwarp -ot Int16 -tr 30 30 {tempoutRaster} {outRaster}'
        os.system(warp_command)

        # Remove the temporary raster file
        if os.path.exists(tempoutRaster):
            os.remove(tempoutRaster)
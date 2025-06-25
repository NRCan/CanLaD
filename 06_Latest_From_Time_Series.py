import numpy as np
import os
import rasterio
import sys
from shapely.geometry import box
import geopandas as gpd
from rasterio.crs import CRS

from rasterio.mask import mask
my_wkt="""PROJCRS["NAD_1983_Canada_Lambert",BASEGEOGCRS["NAD83",DATUM["North American Datum 1983", ELLIPSOID["GRS 1980",6378137,298.257222101004, LENGTHUNIT["metre",1]],ID["EPSG",6269]],PRIMEM["Greenwich",0,ANGLEUNIT["Degree",0.0174532925199433]]],CONVERSION["unnamed", METHOD["Lambert Conic Conformal (2SP)", ID["EPSG",9802]],PARAMETER["Latitude of false origin",0,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8821]], PARAMETER["Longitude of false origin",-95,ANGLEUNIT["Degree",0.0174532925199433], ID["EPSG",8822]], PARAMETER["Latitude of 1st standard parallel",49, ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude of 2nd standard parallel",77,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8824]],PARAMETER["Easting at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8826]],PARAMETER["Northing at false origin",0,LENGTHUNIT["metre",1], ID["EPSG",8827]]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1], LENGTHUNIT["metre",1,ID["EPSG",9001]]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]"""
dst_crs = CRS.from_wkt(my_wkt)

def find_last_events(data, start_year=1985):
    """
    Find the last valid event and its first occurrence year for each pixel in a 3D array.

    Parameters:
    -----------
    data : numpy.ndarray
        3D array of shape (years, height, width) containing event data
        Values of 99 and 100 are considered invalid/no events
    start_year : int
        The starting year for the time series

    Returns:
    --------
    last_event_type : numpy.ndarray
        2D array containing the last valid event type for each pixel
    first_year_last_event : numpy.ndarray
        2D array containing the year when the last event type first occurred
    """

    # Create mask for valid values (not 99 or 100)
    valid_mask = (data != 99) & (data != 100) & (data != 0)

    # Initialize event IDs for each time step
    event_ids = np.zeros_like(data, dtype=int)
    current_event_id = np.zeros(data.shape[1:], dtype=int)

    # Assign event IDs to continuous valid periods
    for t in range(data.shape[0]):
        # New event starts when:
        # 1. Current timestep is valid AND
        # 2. Either it's the first timestep OR value changed from previous timestep
        if t == 0:
            new_event = valid_mask[t]
        else:
            value_changed = (data[t] != data[t - 1]) & valid_mask[t]
            new_event = value_changed

        current_event_id[new_event] += 1
        event_ids[t] = np.where(valid_mask[t], current_event_id, 0)

    # Find the last event ID and its position
    last_event_id = np.max(event_ids, axis=0)
    last_event_layer = np.argmax(event_ids == last_event_id[:, None, ...].transpose(1, 0, 2), axis=0)

    # Get the rows and columns for advanced indexing
    rows, cols = np.indices(last_event_id.shape)

    # Extract the last event type
    last_event_type = data[last_event_layer, rows, cols]

    # Create mask for the last event period
    last_event_mask = (event_ids == last_event_id)

    # Initialize array for first year and last year of last event
    first_year_last_event = np.full(data.shape[1:], np.nan)
    last_year_last_event = np.full(data.shape[1:], np.nan)

    years = np.arange(start_year, start_year + data.shape[0])

    # Find the first year of the last event for each pixel
    for t in range(data.shape[0]):
        last_event_cells = last_event_mask[t]
        # Only update cells that haven't been assigned a year yet
        update_mask = last_event_cells & np.isnan(first_year_last_event)
        first_year_last_event[update_mask] = years[t]

        # Update last year for cells with last event
        last_year_last_event[last_event_cells] = years[t]
        lastyear = years[-1]
        last_year_last_event = np.where((last_year_last_event == lastyear) & (first_year_last_event != lastyear),
                                        np.nan,
                                        last_year_last_event)


    return last_event_type, first_year_last_event, last_year_last_event


# Example usage:
if __name__ == "__main__":


    path_ = sys.argv[1]
    index = sys.argv[2]
    xmin = int(sys.argv[3])
    ymin = int(sys.argv[4])
    xmax = int(sys.argv[5])
    ymax = int(sys.argv[6])
    tif_files = sorted([os.path.join(path_, f) for f in os.listdir(path_) if f.endswith('.tif')])
    print(tif_files)

    outputpath = path_ + '/Clean_latest/'
    os.makedirs(outputpath, exist_ok=True)

    # Open and process rasters within the spatial bounds
    raster_data_list = []

    # Create a polygon from the bounding box
    bounding_box = box(xmin, ymin, xmax, ymax)

    # Convert to GeoJSON format
    geojson_bbox = gpd.GeoSeries([bounding_box]).__geo_interface__["features"][0]["geometry"]

    data_list = []
    transform_list = []
    # Loop through each raster file, apply the mask, and store results
    for path in tif_files:
        with rasterio.open(path) as src:
            # Apply the mask
            masked_data, masked_transform = mask(src, [geojson_bbox], crop=True)
            data_list.append(masked_data)
            transform_list.append(masked_transform)

    combined_rasters = np.array(data_list)
    print('combined_rasters', combined_rasters.shape)
    data = np.squeeze(combined_rasters, axis=1)



    last_types, first_years, last_years = find_last_events(data, start_year=1984)


    outputlastyear = outputpath + str(index) + f"output_top_year.tif"
    outputlastType = outputpath + str(index) + f"output_top_type.tif"
    outputlastyeart2 = outputpath + str(index) + f"output_top_yearT2.tif"


    meta = src.meta.copy()
    meta.update({
        "height": data.shape[1],  # Number of rows (y dimension)
        "width": data.shape[2],  # Number of columns (x dimension)
        "transform": transform_list[0],  # Use the transform from the first masked raster
        "count": 1,  # Number of bands (assuming single-band output)
        "dtype": 'int16',
        "compress": "deflate"
    })

    with rasterio.open(outputlastyear, 'w', **meta) as dst:
        dst.write(first_years, 1)

    with rasterio.open(outputlastType, 'w', **meta) as dst:
        dst.write(last_types , 1)

    with rasterio.open(outputlastyeart2, 'w', **meta) as dst:
        dst.write(last_years, 1)
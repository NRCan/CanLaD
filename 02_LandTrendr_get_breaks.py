from tqdm import tqdm
import os
import sys
import numpy as np
import rasterio
from rasterio.crs import CRS

# Define the projection system (NAD 1983 Canada Lambert)
my_wkt = """PROJCRS["NAD_1983_Canada_Lambert",
    BASEGEOGCRS["NAD83",DATUM["North American Datum 1983",
    ELLIPSOID["GRS 1980",6378137,298.257222101004, 
    LENGTHUNIT["metre",1]],ID["EPSG",6269]],
    PRIMEM["Greenwich",0,ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["unnamed", METHOD["Lambert Conic Conformal (2SP)", ID["EPSG",9802]],
    PARAMETER["Latitude of false origin",0,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8821]], 
    PARAMETER["Longitude of false origin",-95,ANGLEUNIT["Degree",0.0174532925199433], ID["EPSG",8822]], 
    PARAMETER["Latitude of 1st standard parallel",49, ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8823]],
    PARAMETER["Latitude of 2nd standard parallel",77,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8824]],
    PARAMETER["Easting at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8826]],
    PARAMETER["Northing at false origin",0,LENGTHUNIT["metre",1], ID["EPSG",8827]]],
    CS[Cartesian,2],AXIS["(E)",east,ORDER[1], LENGTHUNIT["metre",1,ID["EPSG",9001]]],
    AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]"""

dst_crs = CRS.from_wkt(my_wkt)

# Read command-line arguments
path_out = sys.argv[1]  # Output path
index = sys.argv[2]  # Tile index

# Threshold for disturbance detection
thresh = 100
num_peak = 1  # Number of peaks to consider

# Define VRT file containing all LandTrendr segments
vrtLTD = f'/SR_COL2_summ_Landtrendr/output/tile_{index}/SR_COL2_summ_1st2ndBest_LTD_tile{index}_Breaks.vrt'

# Define output directory and file paths
output_path = f"{path_out}/tile_{index}"
os.makedirs(output_path, exist_ok=True)

output_year = f"{output_path}/last{num_peak}year_LTD_{index}_{thresh}.tif"
output_magn = f"{output_path}/last{num_peak}mag_LTD_{index}_{thresh}.tif"
output_dur = f"{output_path}/last{num_peak}dur_LTD_{index}_{thresh}.tif"
output_start = f"{output_path}/last{num_peak}start_LTD_{index}_{thresh}.tif"
output_reg = f"{output_path}/last{num_peak}reg_LTD_{index}_{thresh}increase100.tif"

def find_negative_peaks(arr, threshold=100):
    """
    Identify significant negative changes (peaks) in a time series array.

    Parameters:
    - arr: 1D NumPy array representing pixel values over time.
    - threshold: The minimum required decrease to be considered a significant peak.

    Returns:
    - Boolean array indicating where significant negative peaks occur.
    """
    # Compute the difference between consecutive years
    diffs = arr[:-1] - arr[1:]

    # Identify indices where the drop exceeds the threshold
    decrease_indices = np.where(diffs >= threshold)[0]

    # Remove consecutive indices, keeping only the first in each sequence
    if len(decrease_indices) > 1:
        non_consecutive_indices = [decrease_indices[0]]  # Always keep the first index
        end_index = []

        for i in range(1, len(decrease_indices)):
            non_consecutive_indices.append(decrease_indices[i])
            end_index.append(decrease_indices[i - 1])

        end_index.append(decrease_indices[-1])
        decrease_indices = np.array(non_consecutive_indices)

    # Create a full-sized boolean array marking the detected peaks
    full_decreases = np.zeros(arr.shape, dtype=bool)
    full_decreases[decrease_indices] = True

    return full_decreases


# Open the VRT file and process raster data
with rasterio.open(vrtLTD) as src:
    data = src.read().astype(float)  # Read raster data as float

    # Extract fit values (change magnitude) and corresponding years
    fit_array = data[0::2, :, :]
    fit_year = data[1::2, :, :]

    # Replace no-data values (-32768) with NaN for processing
    fit_array[fit_array == -32768] = np.nan

    # Initialize arrays to store results
    last_valid_before_nan = np.full((fit_array.shape[1], fit_array.shape[2]), np.nan, dtype=float)
    last_year_before_nan = np.full((fit_array.shape[1], fit_array.shape[2]), np.nan, dtype=int)
    mag_of_last_decrease = np.full((fit_array.shape[1], fit_array.shape[2]), np.nan, dtype=float)
    startyear_of_last_decrease = np.full((fit_array.shape[1], fit_array.shape[2]), np.nan, dtype=float)
    duration_of_last_decrease = np.full((fit_array.shape[1], fit_array.shape[2]), np.nan, dtype=float)

    # Iterate over each pixel (row, col) in the raster
    for row in tqdm(range(fit_array.shape[1]), desc="Processing rows"):
        for col in range(fit_array.shape[2]):
            pixel_values = fit_array[:, row, col]
            negative_peaks = find_negative_peaks(pixel_values, thresh)
            decrease_indices = np.where(negative_peaks)[0]

            if decrease_indices.size > num_peak - 1:
                last_decrease_index = decrease_indices[-num_peak] + 1

                last_valid_before_nan[row, col] = pixel_values[last_decrease_index]
                last_year_before_nan[row, col] = fit_year[last_decrease_index, row, col]  # End year of the  segment
                startyear_of_last_decrease[row, col] = fit_year[last_decrease_index - 1, row, col]  # Start year of the  segment
                duration_of_last_decrease[row, col] = last_year_before_nan[row, col] - startyear_of_last_decrease[row, col] # duration of the  segment

                # Handle edge case for first year of detection
                if fit_year[last_decrease_index, row, col] == 1984.0:
                    last_year_before_nan[row, col] = -32768

                mag_of_last_decrease[row, col] = pixel_values[last_decrease_index - 1] - pixel_values[last_decrease_index] # Magnitude of the segment

    # Update metadata for output GeoTIFFs
    meta = src.meta.copy()
    meta.update({
        'driver': 'GTiff',
        'count': 1,  # Single-band output
        'dtype': 'float32',  # Data type
        'compress': 'lzw'  # Optional compression
    })

    # Write results to GeoTIFF files
    with rasterio.open(output_year, 'w', **meta) as dst:
        dst.write(last_year_before_nan.astype(np.float32), 1)

    with rasterio.open(output_magn, 'w', **meta) as dst:
        dst.write(mag_of_last_decrease.astype(np.float32), 1)

    with rasterio.open(output_dur, 'w', **meta) as dst:
        dst.write(duration_of_last_decrease.astype(np.float32), 1)

    with rasterio.open(output_start, 'w', **meta) as dst:
        dst.write(startyear_of_last_decrease.astype(np.float32), 1)
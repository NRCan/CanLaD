import os
import numpy as np
from osgeo import gdal
import glob
import rasterio
import re
import sys

def create_mask(tif_files, year_index, window_size=5):
    """
    Create a binary mask from a 5-year window sum (or at least 3 years for edges)

    Args:
        tif_files: List of all tif files in chronological order
        year_index: Index of the current year in the list
        window_size: Desired window size (default=5)

    Returns:
        Binary mask after sieve filter
    """
    # Determine the window range (handling edge cases)
    half_window = window_size // 2
    start_idx = max(0, year_index - half_window)
    end_idx = min(len(tif_files), year_index + half_window + 1)

    # Make sure we use at least 3 years for edge cases
    if end_idx - start_idx < 3:
        if year_index < len(tif_files) // 2:
            end_idx = min(len(tif_files), start_idx + 3)
        else:
            start_idx = max(0, end_idx - 3)
    print('end_idx', end_idx)
    print('start_idx', start_idx)

    # Read the first file to get dimensions
    ds = gdal.Open(tif_files[start_idx])
    rows = ds.RasterYSize
    cols = ds.RasterXSize
    geo_transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    ds = None  # Close dataset

    # Sum the values in the window
    sum_array = np.zeros((rows, cols), dtype=np.float32)
    print('sum_array',(sum_array.shape))
    for i in range(start_idx, end_idx):
        print(tif_files[i])

        with rasterio.open(tif_files[i]) as src:
            data = src.read(1)

        sum_array += data

    # Create binary mask (change/no change)
    binary_mask = np.zeros_like(sum_array, dtype=np.uint8)
    binary_mask[sum_array != 0] = 1

    # Write binary mask to temporary file for sieve filter
    driver = gdal.GetDriverByName('GTiff')
    temp_file = "/temp_mask/temp_mask"+str(year_index)+".tif"
    ds_temp = driver.Create(temp_file, cols, rows, 1, gdal.GDT_Byte)
    ds_temp.SetGeoTransform(geo_transform)
    ds_temp.SetProjection(projection)
    band_temp = ds_temp.GetRasterBand(1)

    # Convert numpy array to bytes buffer
    # Make sure binary_mask is the correct data type (Byte/uint8 in this case)
    binary_mask = binary_mask.astype(np.uint8)
    data_buffer = binary_mask.tobytes()

    # Write using WriteRaster instead of WriteArray
    band_temp.WriteRaster(0, 0, cols, rows, data_buffer)

    # Flush cache to ensure data is written
    ds_temp.FlushCache()
    band_temp.FlushCache()


    # Apply sieve filter (minimum connected pixels = 12)
    gdal.SieveFilter(srcBand=band_temp, maskBand=None, threshold=12, connectedness=4, dstBand=band_temp)

    return temp_file


def apply_mask_to_disturbance(disturbance_file, mask_file):
    """
    Apply mask to disturbance raster
    When pixel in disturbance != 0 and mask == 0, set to 0
    """


    with rasterio.open(disturbance_file) as src:
        disturbance_data = src.read(1)

    with rasterio.open(mask_file) as src:
        maskarray = src.read(1)


    print('mask', maskarray.shape)


    # Apply mask logic
    # When disturbance pixel != 0 AND mask pixel == 0, set to 0
    masked_data = disturbance_data.copy()
    masked_data[(disturbance_data != 0) & (maskarray == 0)] = 0

    # Write back to the same file
    with rasterio.open(disturbance_file, 'r+') as dst:
        dst.write(masked_data, 1)  # Write to band 1

    return masked_data


def process_disturbance_time_series(folder_path, years_to_process=None, output_folder=None):
    """
    Process disturbance TIF files for specific years

    Args:
        folder_path: Path to folder containing TIF files
        years_to_process: List of years to process (e.g., [2010, 2015, 2020])
                         If None, process all years
        output_folder: Output folder path. If None, creates a 'masked' folder
    """
    # Create output folder
    if output_folder is None:
        output_folder = os.path.join(folder_path, "masked")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all TIF files and sort them
    all_tif_files = sorted(glob.glob(os.path.join(folder_path, "*.tif")))
    print(all_tif_files)
    if len(all_tif_files) == 0:
        print(f"No TIF files found in {folder_path}")
        return

    # Filter files by year if years_to_process is specified
    if years_to_process:
        tif_files = []
        for file_path in all_tif_files:
            file_name = os.path.basename(file_path)
            # Extract year from filename (assuming format like 'something_YYYY.tif')
            # Adjust this regex pattern to match your specific filename format
            year_match = re.search(r'_(\d{4})\.tif$', file_name)
            if year_match and int(year_match.group(1)) in years_to_process:
                tif_files.append(file_path)

        #path.join(dirpath, f)

        print(f"Found {len(tif_files)} TIF files matching the specified years out of {len(all_tif_files)} total files.")
    else:
        tif_files = all_tif_files
        print(f"Processing all {len(tif_files)} TIF files found.")

    if len(tif_files) == 0:
        print("No files to process after filtering by year.")
        return

    # Process each selected file
    for i, tif_file in enumerate(tif_files):
        file_name = os.path.basename(tif_file)
        print(f"Processing {file_name} ({i + 1}/{len(tif_files)})")

        # Create mask for this year
        # Pass all files and current index to create_mask
        mask = create_mask(all_tif_files, all_tif_files.index(tif_file))

        # Apply mask to the disturbance raster
        output_file = os.path.join(output_folder, file_name)


        # Apply mask to the output file
        apply_mask_to_disturbance(output_file, mask)
        print(f"  Saved masked result to {output_file}")

    print("Processing complete!")


# Example usage
if __name__ == "__main__":
    # folder path

    # Specify which years to process
    # In Job
    folder_path = sys.argv[1]
    date = sys.argv[2]
    years_to_process=[int(date)]

    # Process only the specified years
    process_disturbance_time_series(folder_path, years_to_process)
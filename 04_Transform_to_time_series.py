import numpy as np
import rasterio
from rasterio.transform import from_origin
import os
import fnmatch
import pandas as pd
from tqdm import tqdm
import sys

# Define time range for analysis
start_year, end_year = 1984, 2024
total_years = end_year - start_year + 1

# Define directories for input datasets
Start_dir = '/output_start/'  # Disturbance start year
dir = '/ResultatTempCNN/output/'  # Results Classification


# Retrieve user-defined parameters from command line arguments
path_out = sys.argv[1]  # Output directory
index = sys.argv[2]  # Tile index
dirout = f"{path_out}/{index}/"  # Output directory for results

# Ensure output directory exists
os.makedirs(dirout, exist_ok=True)

# Define directories for different stages of processing
dir_start1 = f"{dir}start1/{index}/"
dir_start2 = f"{dir}start2/{index}/"
dir_start3 = f"{dir}start3/{index}/"
dir_start4 = f"{dir}start4/{index}/"
dir_start5 = f"{dir}start5/{index}/"

# Define file paths for each stage of the disturbance detection process
type_start_files = [f"{dir_start}{index}/result_{index}_type.tif" for dir_start in
                    [dir_start1, dir_start2, dir_start3, dir_start4, dir_start5]]

# Define file paths for LandTrendr disturbance detection results
Start_LTD_files = [f"{Start_dir}tile_{index}/last{i}start_LTD_{index}_100.tif" for i in range(1, 6)]
regeneration_files = [f"{Start_dir}tile_{index}/last{i}year_LTD_{index}_100.tif" for i in range(1, 6)]

# List of final datasets for processing
FINALYEARLIST = [Start_LTD_files[i] if i % 2 == 0 else regeneration_files[i] for i in range(10)]
FINALTypeLIST = [type_start_files[i] if i % 2 == 0 else regeneration_files[i] for i in range(10)]


def reorder_reg_values(yeardata, valuedata):
    """
    Reorder 'reg' values based on corresponding 'start' values.

    Parameters:
    - yeardata (numpy.ndarray): 1D array of start and end years for events.
    - valuedata (numpy.ndarray): 1D array of corresponding event values.

    Returns:
    - tuple: Reordered year and value arrays.
    """
    events = yeardata.reshape(5, 2)  # Reshape into (5 events, 2 values each)
    events_type = valuedata.reshape(5, 2)

    starts = events[:, 0]
    regs = events[:, 1]
    regs_type = events_type[:, 1]

    reordered_regs = np.full_like(regs, 99)  # Initialize with no-data values
    reordered_regs_type = np.full_like(regs_type, 99)

    # Sort events by 'start' year in descending order
    sorted_indices = np.argsort(-starts)

    for i in sorted_indices:
        start = starts[i]

        if start in [99, 100]:  # Ignore no-data values
            continue

        valid_indices = np.where((regs >= start))[0]
        if valid_indices.size > 0:
            closest_index = valid_indices[0]  # Select the closest valid reg value
            reordered_regs[i] = regs[closest_index]
            reordered_regs_type[i] = regs_type[closest_index]

    return events.flatten(), events_type.flatten()


def transform_to_time_series(exyear, extype, start_year=1984, end_year=2024):
    """
    Convert breakpoint data to a full time series.

    Parameters:
    - exyear (array): Breakpoint years.
    - extype (array): Corresponding event types.
    - start_year (int): First year in the time series.
    - end_year (int): Last year in the time series.

    Returns:
    - np.ndarray: Full time series array.
    """
    num_years = end_year - start_year + 1
    time_series = np.full(num_years, 99, dtype=np.float32)

    exyear, extype = reorder_reg_values(exyear, extype)

    for i in range(0, len(exyear), 1):  # Iterate

        start = exyear[i]
        end = exyear[i + 1] if i + 1 < len(exyear) else 0
        value = extype[i] if i < len(extype) else 0
        valid_condition = ((start != 0) & (end != 0) &
                           (start != 99) & (start != 100) &
                           (end != 99) & (end != 100))

        # Calculate duration: dur = start - end + 1 if valid, else 0
        dur = np.where(valid_condition, end - start + 1, 0)
        dur = 9 if dur == 10 else dur

        # condition to assign the disturbance year based on the end of segment for fire and harvesting
        if value in [1, 14] and dur >= 3:
            start = end



        # Skip invalid or missing data
        if start == 99 or start < start_year or value is None or np.isnan(value):
            continue
        # Skip invalid or missing data
        if end == -32768.0:
            continue
        # Skip invalid or missing data
        if start != 99 and value == 100 and (start < exyear[i - 1] or start < exyear[i - 2]):
            continue



        # Verify value and year of earlier index, if nothing before=> continue
        if start != 99:
            all_earlier_start_are_99 = False
        if value != 99:
            all_earlier_value_are_99 = False
        if start != 99 and value == 100 and all_earlier_value_are_99:
            continue
        if start != 99 and value == 100 and all_earlier_start_are_99:
            continue

        # Conditionally adjust start_idx to select date of the end of the disturbance and dill it with 100 after
        if value == 100:
                       start_idx = int(max(start - start_year + 1, 0))  # Incremented start index
        else:
            start_idx = int(max(start - start_year, 0))  # Default start index

            # Apply conditions based on the type of classes (value)
        if value in [0, 3, 6, 8, 9, 12, 13, 7, 11]:  # Forest, NoForest, Agri, Urban, Water, Rock, Recovery class are== nochange:
            value = 99  # Change to 99
        elif value in [14]:  # change pest_fire to fire
            value = 1  # Ignore `type=100` for these values
        elif value in [1, 2, 5, 10]:  # Fire, harvesting,  windthrow, ad dam only 1 years:
            extype[i + 1] = 100
        if value in [1, 2, 5, 10]:  # Fire, harvesting,  windthrow, ad dam only 1 years:
            end_idx = int(start + 1 - start_year)


        else:
            # Assign values to the time series
            # end_idx = int(min((end - start_year + 1) if end and end != 99 else num_years, num_years))
            end_idx = int(end_year - start_year + 1)

        #Reclass the type code
        reclass_map = {99: 99, 100: 100, 1: 1, 2: 2, 4: 99, 5: 3, 10: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 8}
        if value in reclass_map:
            value = reclass_map[value]


        # Calculate duration for value transformation (but not for 99 or 100)
        if value not in [99, 100]:
            value = value * 10 + dur


        time_series[start_idx:end_idx] = value

        # if the same value appears twice separated by less than 5 years,
        #     change the first occurrence to 100.
    modified = time_series.copy()
    gap_threshold = 5

    # Iterate through the array
    for i in range(len(time_series)):
        current_value = time_series[i]

        # Skip if current value is 99 or 100 (assuming these are special values)
        if current_value in [99, 100]:
            continue

        first_digit = int(str(int(current_value))[0]) if current_value != 0 else 0
        if first_digit not in [1, 2, 3, 4]:
            continue

        # Look for the same value in the next gap_threshold positions
        for j in range(i + 1, min(i + gap_threshold + 1, len(time_series))):
            if time_series[j] in [99, 100]:
                continue

            current_digit = int(str(int(time_series[j]))[0])
            check_digit = int(str(int(current_value))[0])

            if current_digit == check_digit:

                # Found same value within gap_threshold, change first occurrence to 100
                modified[i] = 99
                break


    return time_series


def transform_band_to_binary(time_series, band_index, non_nan_value=100, nan_value=0):
    """
    Convert a specific band to binary format.

    Parameters:
    - time_series (numpy.ndarray): 3D array of time series data.
    - band_index (int): Index of the band to process.
    - non_nan_value (int): Value for non-NaN pixels.
    - nan_value (int): Value for NaN pixels.

    Returns:
    - numpy.ndarray: Transformed time series.
    """
    band = time_series[band_index]
    band[band == -32768] = 99  # Replace no-data values

    band[~np.isnan(band)] = non_nan_value
    band[np.isnan(band)] = nan_value
    time_series[band_index] = band
    return time_series


with rasterio.open(FINALTypeLIST[0], "r+") as src:
    meta = src.meta
    height, width = src.shape
    time_series = np.full((total_years, height, width), 99, dtype=np.int8)

    # Load raster data
    year_data = np.array([rasterio.open(path).read(1) for path in FINALYEARLIST])
    year_data = np.nan_to_num(year_data, nan=99)

    type_data = np.array([rasterio.open(path).read(1) for path in FINALTypeLIST])
    for band_index in [-1, 1, 3, 5, 7]:
        type_data = transform_band_to_binary(type_data, band_index, non_nan_value=100, nan_value=99)

    # Process each pixel
    for row in tqdm(range(year_data.shape[1])):
        for col in range(year_data.shape[2]):
            year_values = year_data[:, row, col]
            type_values = type_data[:, row, col]

            # StartIndice need +1 (If from LTD)
            indices = [0, 2, 4, 6, 8]
            year_values[indices] += 1

            result1d = transform_to_time_series(year_values, type_values)


            time_series[:, row, col] = result1d

    for i in range(total_years):
        output_filename = f"{dirout}timeserie_{start_year + i}.tif"
        with rasterio.open(output_filename, "w", **meta) as dst:
            dst.write(time_series[i, :, :], 1)

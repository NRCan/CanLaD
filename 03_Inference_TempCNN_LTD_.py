import os
import rioxarray
from utils.Final_tiff import makemap
from utils.Dataset10y_SW import Dataset_inf as myDataset
from torch.utils.data.dataset import Subset
import torch
import pandas as pd
import sys
import numpy as np
from tqdm import tqdm




def inference(pytorch_network, loader):
    """
    This function transfers the neural network to the right device,
    apply the network to the data and get the probability result.

    Args:
        pytorch_network (torch.nn.Module): The neural network to apply.
        loader (torch.utils.data.DataLoader): The DataLoader to infer on.

    Returns:
        Array with probability for all classes and targets with x,y coordinate.

    """
    cx = []
    cy = []
    cnan=[]
    cyear=[]

    predstype = []
    predsdate = []
    predsdateF = []

    pytorch_network.eval()
    with torch.no_grad():
        for batch, pix_nan, pix_x, pix_y, pix_year in loader:
            x = batch['sequence']

            # Transfer batch on GPU if needed.
            x = x.to(device)

            # Get the type prediction of the network and reshape it
            y_pred_type = pytorch_network(x)['type']
            predtype = torch.argmax(y_pred_type, 1)
            predtype = predtype.view(-1).cpu().numpy()
            predtype = np.reshape(predtype, (len(predtype), 1))



            # Organize the output
            for i in range(len(predtype)):
                predstype.append(predtype[i])


                cx.append(pix_x[i])
                cy.append(pix_y[i])
                cnan.append(pix_nan[i])
                cyear.append(pix_year[i])

        final_result = np.column_stack((cx, cy, cnan, cyear, predstype))

    return final_result



###############################################################
####### Read image with slidding windows

def load_images_by_year_band_winter(img_path, img_path_w, mask_path, years, bands):
    """
    Load images automatically for each year and band from summer and winter folders.

    Parameters:
    - img_path: Path to summer images.
    - img_path_w: Path to winter images.
    - mask_path: Path to the masks.
    - years: List of years to load (e.g., [1980, 1981, ..., 2020]).
    - bands: List of band numbers (e.g., [1, 2, 3, 4, 5, 6]).

    Returns:
    - images_by_year_band: Dictionary where the key is the year, and the value is a nested dictionary
      with "summer" and "winter" entries for each band.
    """
    images_by_year_band = {}

    for year in tqdm(years):
        images_by_year_band[year] = {"summer": {}, "winter": {}}  # Initialize summer and winter dictionaries

        for band in bands:
            # Summer image path
            img_name = f'SR_L5789_CS_{year}_B{band}.tif'
            img_path_summer = os.path.join(img_path, img_name)


            # Load summer image if it exists
            if os.path.exists(img_path_summer):
                rds_summer = rioxarray.open_rasterio(img_path_summer).sel(x=slice(x, x + w_h), y=slice(y, y - w_w))
                images_by_year_band[year]["summer"][band] = rds_summer.values
            else:
                print(f"Summer file {img_name} does not exist, skipping...")


    return images_by_year_band

def load_images_by_year_band(img_path, mask_path, years, bands):
    """
    Load images automatically for each year and band from a folder.

    Parameters:
    - folder_path: The path to the folder containing the images.
    - years: List of years to load (e.g., [1980, 1981, ..., 2020]).
    - bands: List of band numbers (e.g., [1, 2, 3, 4, 5, 6]).

    Returns:
    - images_by_year_band: Dictionary where the key is the year and the value is another
      dictionary containing band images for that year.
    """
    images_by_year_band = {}

    # Loop over each year and band
    for year in tqdm(years):
        images_by_year_band[year] = {}  # Initialize a dictionary for each year
        #mask_name = f'xgb_cloudShadowMask_{year}.tif'
        #mask_path_ = os.path.join(mask_path, mask_name)


        for band in bands:
            # Construct the file name based on the naming pattern
            img_name = f'SR_L5789_CS_{year}_B{band}.tif'
            img_path_ = os.path.join(img_path, img_name)

            # Check if the file exists
            if os.path.exists(img_path):
                # Load the image
                rds = rioxarray.open_rasterio(img_path_).sel(x=slice(x, x + w_h), y=slice(y, y - w_w))
                #mask = rioxarray.open_rasterio(mask_path_).sel(x=slice(x, x + w_h), y=slice(y, y - w_w))
                #masked_rds = xr.where(mask, -32768, rds)

                image_array = rds.values
                images_by_year_band[year][band] = image_array
            else:
                print(f"File {img_name} does not exist, skipping...")

    return images_by_year_band




def adjust_reference_year(year_disturbance, mag, min_year=1985, max_year=2024, required_years=10):
    """
    Adjust the reference year based on the LTD disturbance year.

    Parameters:
    - year_disturbance: a disturbance year.
    - min_year: The minimum year allowed (default is 1986).
    - max_year: The maximum year allowed (default is 2024).
    - required_years: The number of years required (default is 10).

    Returns:
    - adjusted_year: adjusted reference year.
    """
    if mag > 10:
        return np.nan
    if year_disturbance < min_year:
        return np.nan

    if min_year <= year_disturbance <= 2016:
        #print('year_disturbance', year_disturbance)
        adjusted_year = year_disturbance - 1
    elif 2017 <= year_disturbance <= max_year:
        adjusted_year = max_year - required_years + 1
    else:
        adjusted_year = year_disturbance

    return adjusted_year



def create_composite_vectorized(year_image, mag, images_by_year, years, num_years=10, num_bands=6):
    """
    Create a composite image based on a reference year image and include 10 successive years and all bands.

    Parameters:
    - year_image: 2D numpy array where each pixel contains a reference year.
    - images_by_year: Dictionary where the key is the year and the value is a dictionary of 2D arrays for each band.
                      Example: {1985: {1: band1_image, 2: band2_image, ..., 6: band6_image}}
    - years: List of years corresponding to the images_by_year keys.
    - num_years: Number of successive years to consider for the composite (default is 10).
    - num_bands: Number of bands per year (default is 6).

    Returns:
    - composite_image: 3D numpy array where the first dimension corresponds to bands,
                       and each pixel is the composite value from the 10 successive years.
    """

    _, height, width = year_image.shape
    composite_image = np.full((num_bands * num_years + 1, height, width), -32768, dtype=np.float32)

    # Convert images_by_year dictionary to stacked arrays
    # Stacking years and bands to create an array of shape (total_years, num_bands, height, width)
    stacked_images = np.stack(
        [
            np.squeeze(images_by_year[year][band])
            for year in years
            for band in (1, 2, 3, 4, 5, 7)
        ],
        axis=0
    )  # shape: (num_years * num_bands, height, width)

    # Create a lookup for the reference years
    year_to_index = {year: idx for idx, year in enumerate(years)}

    # Calculate the real x and y coordinates using broadcasting
    real_x = np.tile(np.arange(xmin + 15, xmax, 30), (height, 1)).astype(np.float32)
    real_y = np.tile(np.arange(ymax - 15, ymax - height * 30, -30)[:, None], (1, width)).astype(np.float32)


    for y in tqdm(range(height)):
        for x in range(width):
            reference_year = adjust_reference_year(year_image[0, y, x], mag[0, y, x])

            # Check if reference year exists
            while reference_year in year_to_index:
                start_year_idx = year_to_index[reference_year]
                start_idx = start_year_idx * num_bands
                end_idx = start_idx + (num_years * num_bands)



                # Extract the pixel values for the current reference year and successive years
                pixel_values = stacked_images[start_idx:end_idx, y, x]

                if pixel_values[0] != -32768:
                    composite_image[:num_years * num_bands, y, x] = pixel_values
                    composite_image[-1, y, x] = reference_year
                    break

                # Decrement the reference year if the first year's value is invalid
                reference_year -= 1



    # Concatenate real_x and real_y into the composite image
    composite_image_f = np.concatenate(
        [composite_image, np.expand_dims(real_x, axis=0), np.expand_dims(real_y, axis=0)], axis=0
    )

    return composite_image_f

def create_composite(year_image, mag, images_by_year, years, num_years=10, num_bands=6):
    """
    Create a composite image based on a reference year image and include 10 successive years and all bands.

    Parameters:
    - year_image: 2D numpy array where each pixel contains a reference year.
    - images_by_year: Dictionary where the key is the year and the value is a dictionary of 2D arrays for each band.
                      Example: {1985: {1: band1_image, 2: band2_image, ..., 6: band6_image}}
    - years: List of years corresponding to the images_by_year keys.
    - num_years: Number of successive years to consider for the composite (default is 10).
    - num_bands: Number of bands per year (default is 6).

    Returns:
    - composite_image: 3D numpy array where the first dimension corresponds to bands,
                       and each pixel is the composite value from the 10 successive years.
    """

    # Initialize an empty array for the composite image
    # The output will be of shape (num_bands, height, width), same as one band's image size
    _, height, width = year_image.shape
    composite_image = np.full((num_bands * num_years +1, height, width), -32768, dtype=np.float32)


    # Loop over all pixels in the year_image
    for y in tqdm(range(height)):
        for x in range(width):

            # Get the reference year for the current pixel
            reference_year = adjust_reference_year(year_image[0,y, x], mag[0,y, x])

            # Find the 10 successive years starting from the reference year
            start_year_index = years.index(reference_year) if reference_year in years else None
            composite_layer_index = 0
            if start_year_index is not None:
                # For each band, compute the composite from successive years
                for i in range(num_years):
                    pixel_values = []
                    year_index = start_year_index + i
                    if year_index < len(years):
                        year = years[year_index]

                        for band in (1, 2, 3, 4, 5, 7 ):

                            composite_image[i * num_bands + band - 1, y, x] = images_by_year[year][band][0, y, x]

                    # Compute the composite for this pixel and band
                composite_image[num_bands * num_years, y, x] = reference_year

                composite_image_f = np.concatenate((composite_image, np.expand_dims(real_x, axis=0), np.expand_dims(real_y, axis=0)), axis=0)

    return composite_image_f


def array_to_dataframe(array, band_names=None):
    """
    Transforms a 3D array into a DataFrame with columns for coordinates (x, y) and band values.

    Parameters:
    - array: 3D numpy array with shape (bands, rows, columns).
    - band_names: Optional list of names for the bands (e.g., ['band1', 'band2', ...]).
      If None, default band names will be used.

    Returns:
    - df: Pandas DataFrame with columns 'x', 'y', and the band values (e.g., 'band1_y1', 'band2_y1', ...).
    """

    # Get dimensions of the array
    bands, rows, cols = array.shape

    # Generate x and y coordinates
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Reshape the array
    reshaped_array = array.reshape(bands, -1).T  # Shape: (rows*cols, bands)

    # Create column names for bands if not provided
    if band_names is None:
        band_names = [f'band{i + 1}' for i in range(bands)]


    # Create a DataFrame with the reshaped array and coordinates
    df = pd.DataFrame(reshaped_array, columns=band_names)


    return df


if __name__ == "__main__":
    # Read input arguments from command line
    path_ = sys.argv[1]  # Output directory
    index = sys.argv[2]  # Tile index
    xmin = int(sys.argv[3])  # Bounding box (min X)
    ymin = int(sys.argv[4])  # Bounding box (min Y)
    xmax = int(sys.argv[5])  # Bounding box (max X)
    ymax = int(sys.argv[6])  # Bounding box (max Y)

    # Define paths for required datasets
    mask_path_ = '/Landsat/20240930_SR_COL2_summ_gapfill_1st2ndBest_with_shadow_gee/xgb_cloudShadowMask'  # Landsat time-series mask
    SummerImg_path = '/Landsat/SR_COL2_summ_gapfill_1st2ndBest_without_shadow_gee/'  # Landsat time-series images

    # LandTrendr disturbance detection results
    LTD_lastest_year = '/SegLTD/output_start_mosaic/Mosaicresult1_start.tif'
    LTD_lastest_dur = '/SegLTD/output_start_mosaic/Mosaicresult1_dur.tif'

    # Define output path
    outputpath = f"{path_}/start5/{index}/"
    os.makedirs(outputpath, exist_ok=True)  # Ensure directory exists

    # Define image and model paths
    IMG_PATH = SummerImg_path + '*B[123457]*.tif'  # Landsat band selection
    model_path = '/TempCNN_CAN_30_s2.pt'  # Deep learning model

    # Set device for PyTorch (GPU if available, otherwise CPU)
    cuda_device = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained deep learning model
    model = torch.load(model_path, map_location=device)
    model.eval()  # Set model to evaluation mode

    # Load Landsat images grouped by year and band
    years = list(range(1984, 2024 + 1))
    images_by_year_band = load_images_by_year_band(SummerImg_path, mask_path_, years, bands)

    # Load LandTrendr results for disturbance year and magnitude
    rds = rioxarray.open_rasterio(LTD_lastest_year).sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
    year_image = rds.values

    rds_mag = rioxarray.open_rasterio(LTD_lastest_dur).sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
    mag_image = rds_mag.values

    # Create time-series composite for inference
    composite = create_composite_vectorized(year_image, mag_image, images_by_year_band, years)

    # Define band names for the composite dataset
    band_names = [f'band{band}_y{year}' for year in range(1, 11) for band in (1, 2, 3, 4, 5, 7)]

    # Add reference year and coordinate columns
    band_names.extend(["refyear", "x", "y"])

    # Convert 3D array to Pandas DataFrame
    df = array_to_dataframe(composite, band_names)

    # Count missing values (-32768) per row
    df['NaN_Count'] = (df == -32768).sum(axis=1)

    # Convert DataFrame into dataset for model inference
    test_dataset = myDataset(df)

    # Create DataLoader for batch inference
    batch_size = 8072
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Perform inference using the TempCNN model
    result = inference(model, test_loader)

    # Define column names for output DataFrame
    allcolumnlist = ('x', 'y', 'NaN_Count', 'refyear', 'type')

    # Store model predictions in DataFrame
    allresultdf = pd.DataFrame(result, columns=allcolumnlist)

    # Group by spatial coordinates and take the mean value
    allresultdf_ = allresultdf.groupby(['x', 'y'], as_index=False).mean()

    # Convert 'type' column to integer after rounding
    allresultdf_['type'] = allresultdf_['type'].round(0).astype(int)



    # Handle no disturbance pixel

    allresultdf_['type'] = np.where(allresultdf_['refyear'] == -32768, 99, allresultdf_['type'])

    # Convert the final DataFrame to a raster (.tif) map
    makemap(allresultdf_, str(index), outputpath)

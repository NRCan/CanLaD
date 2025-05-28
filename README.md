# CanLaD - Canada Landsat Disturbance – Forest Insect Update

This repository contains the code used to generate the results presented in the paper:  
**_Remote Sensing-Based Interannual Monitoring of Major Insect Disturbances in Canadian Forests_** (in review).  
We used **LandTrendr** and the **TempCNN** model to detect forest disturbances across Canada from 1985 to 2024.

👉 Explore the [interactive demo](https://gcpm041u-lemur.projects.earthengine.app/view/canladpest)

---

## Table of Contents

1. [Data Availability](#data-availability)  
2. [Methodology](#methodology)  
3. [Citation](#citation)

---

## Data Availability

Below is a summary of all the published datasets:

- **Canada Landsat Disturbance (CanLaD) – 2025 Forest Pest Update**  
  [Open.canada.ca](***)

- **Disturbance Detection Prior to 1984**  
  [Open.canada.ca](https://doi.org/10.3929/ethz-b-000609845)

- **Canada Landsat Disturbance (CanLaD) – 2017**  
  [Open.canada.ca](https://open.canada.ca/data/en/dataset/add1346b-f632-4eb9-a83d-a662b38655ad)

- **Example Training Dataset**  
  [GitHub](https://github.com/Patawaitte/FoDiM/tree/main/Dataset)

---

## Methodology

1. **Train the TempCNN model**  
   Use `01_Main_Train_Kfold.py`  
   👉 [Example training dataset](https://github.com/Patawaitte/FoDiM/tree/main/Dataset)

2. **Detect disturbance breaks with LandTrendr**  
   Use `02_LandTrendr_get_breaks.py`

3. **Apply TempCNN model to each break**  
   Run `03_Inference_TempCNN_LTD.py`

4. **Generate annual disturbance maps**  
   Convert outputs using `04_Transform_to_time_series.py`

5. **Apply cleaning with 12-pixel sieve filter**  
   Use `05_sieve_time_series.py`

6. **Extract the latest disturbance type and year**  
   Use `06_Latest_From_Time_Series.py`

🖥️ **Note:**  
The entire workflow was executed using the Government of Canada’s High Performance Computing (HPC) service.  
The TempCNN model has a small memory footprint (<1 MB), making CPU-based inference efficient.  
Canada was divided into ~2,000 tiles (10,000 km² each) for parallel processing.  
Full processing time: ~2 days.

---

## Citation

If you use this code or dataset, please cite the associated article (in review):

```bibtex
@article{perbet_canlad_2025,
  author    = {Pauline Perbet et al.},
  title     = {Remote Sensing-Based Interannual Monitoring of Major Insect Disturbances in Canadian Forests},
  journal   = {In Review},
  year      = {2025}
}

"""
Environmental Variables (MUCAPE, MULCL, RH36mean, LR36) at Lightning/OT Detection Points

Purpose:
    For each categorized lightning/OT CSV (land/ocean, LS-only/LS+OT/OT-only), extract
    point-level values of MUCAPE, MULCL, RH36mean, and LR36 from BARRA-R2 reanalysis at
    the nearest grid point/time to each detection, restricted to warm-season months
    (Oct-Apr). RH36mean and LR36 were added later for statistical (t-test) comparisons
    between categories, alongside the originally computed MUCAPE and MULCL.


Input:
    - Categorized lightning/OT point CSVs (from 06_categorical_lightning_OT_dilation.ipynb):
      ls_only_land_3d.csv, ls_only_ocean_3d.csv, ls_ot_land_3d.csv, ls_ot_ocean_3d.csv,
      ot_only_land_3d.csv, ot_only_ocean_3d.csv
    - BARRA-R2 hourly reanalysis: MUCAPE, MULCL, RH36mean, LR36
    - Region: tropical Australia (lat: -20 to -10, lon: 120 to 148)
    - Warm season: October-April, years 2016-2024

Processing:
    1. For each categorized CSV and each year, load MUCAPE, MULCL, RH36mean, and LR36
       fields for the full year, merge them into one dataset.
    2. Extract values at the nearest grid point/time to each detection point in the
       region for that year, restricted to warm-season months.
    3. Concatenate all years into one point-level dataframe per category and save to
       feather.

Output:
    Per-category point-level environmental variable dataframes:
    /g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/
    1_env_variables_<csv_file>.feather

Usage:
    Run: python env_variables_at_detection_points.py
"""

import pandas as pd
import xarray as xr
# Standard libraries
import os
import glob

import pyarrow

from dask.distributed import Client

# Initialize Dask client with one thread per worker
# why this, since I added here all the variables instead of just point in the view of doing t-test. and mucape > 0 is done in later stage on csv files generated instead
# of here, it will over burden the code, so, it has been not done , all though name suggest otherwise.


def get_files(path: str, start: str, end: str): #, warm_months) -> list[str]:
    months = pd.date_range(start=start, end=end, freq='MS').strftime('%Y%m')
    # months = months[months.month.isin(warm_months)].strftime('%Y%m')
    files  = sorted(f for m in months for f in glob.glob(os.path.join(path, f'*-{m}.nc')))
    return files

def preprocess_fn(ds):
    PAD = 2
    ds = (
        ds.drop_vars("crs", errors="ignore")
          .sel(
              lat=slice(LAT_MIN - PAD, LAT_MAX + PAD),
              lon=slice(LON_MIN - PAD, LON_MAX + PAD),
              # time=slice(, END_DATE),    # ← add back here
          )
    )
    return ds


def select_point_based_data(data, variable,year, warm_months=(10, 11, 12, 1, 2, 3, 4)):
    """
    Select nearest grid-point data for OT/lightning points.
    """

    # Copy to avoid modifying original dataframe
    variable = variable.copy()

    # Convert time once
    variable['time'] = pd.to_datetime(variable['time'])

    # Single combined mask (faster + cleaner)
    mask = (
        (variable['lat'].between(LAT_MIN, LAT_MAX)) &
        (variable['lon'].between(LON_MIN, LON_MAX)) &
        (variable['time'].dt.year == year) #&
        (variable['time'].dt.month.isin(warm_months))
    )

    variable = variable.loc[mask]

    # Reduce precision slightly for cleaner nearest-neighbour matching
    variable['lat'] = variable['lat'].round(4)
    variable['lon'] = variable['lon'].round(4)

    # Convert directly to NumPy arrays (lighter than pandas Series)
    da_lat = xr.DataArray(variable['lat'].to_numpy(), dims='points')
    da_lon = xr.DataArray(variable['lon'].to_numpy(), dims='points')
    da_time = xr.DataArray(variable['time'].to_numpy(), dims='points')

    # Nearest selection
    data_select = data.sel(
        lat=da_lat,
        lon=da_lon,
        time=da_time,
        method='nearest'
    )

    return data_select



# list_dir_1 = ['MUCAPE'] 
LAT_MIN, LAT_MAX = -20, -10
LON_MIN, LON_MAX = 120, 148
DATA_DIR   = '/g/data/ob53/BARRA2/output/reanalysis/AUST-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/'
PAD = 2
# start = "2016-01-01"
# end   = "2024-12-31"
years = range(2016, 2025)
csv_files = ["ls_only_land_3d.csv","ls_only_ocean_3d.csv","ls_ot_land_3d.csv","ls_ot_ocean_3d.csv","ot_only_land_3d.csv","ot_only_ocean_3d.csv"]
# csv_files = ["ls_ot_ocean_3d.csv","ot_only_land_3d.csv","ot_only_ocean_3d.csv"]
CSV_DIR = "/g/data/if69/tp4064/Project_1/categorical_position/Dialation/"


def main():
    client = Client(threads_per_worker=1)

    for f in csv_files:
    
        print('yes')
    
        all_data = []
    
        df = pd.read_csv(os.path.join(CSV_DIR + f))
    
        for year in years:
    
            
            start = f"{year}-01-01"
            end   = f"{year}-12-31"
    
            print(start, end)
            
            data_mucape = (xr.open_mfdataset(
                    get_files(
                        os.path.join(DATA_DIR, 'MUCAPE', 'latest/'),
                        start,
                        end,
                        ),
                    chunks={},  # or {'time': 240} later
                    combine='by_coords',
                    engine="netcdf4",
                    parallel=True,
                    preprocess=preprocess_fn
                ))
    
            data_mulcl = (xr.open_mfdataset(
                            get_files(
                                os.path.join(DATA_DIR, 'MULCL', 'latest/'),
                                start,
                                end,
                                ),
                            chunks={},  # or {'time': 240} later
                            combine='by_coords',
                            engine="netcdf4",
                            parallel=True,
                            preprocess=preprocess_fn
                        ))
    
            # these two(data_rh36, data_lr36) were not in the inital calculations, these has been added into the code, for statistical t test 
            data_rh36mean = (xr.open_mfdataset(
                            get_files(
                                os.path.join(DATA_DIR, 'RH36mean', 'latest/'),
                                start,
                                end,
                                ),
                            chunks={},  # or {'time': 240} later
                            combine='by_coords',
                            engine="netcdf4",
                            parallel=True,
                            preprocess=preprocess_fn
                        ))
            data_lr36 = (xr.open_mfdataset(
                            get_files(
                                os.path.join(DATA_DIR, 'LR36', 'latest/'),
                                start,
                                end,
                                ),
                            chunks={},  # or {'time': 240} later
                            combine='by_coords',
                            engine="netcdf4",
                            parallel=True,
                            preprocess=preprocess_fn
                        ))
            
            data = xr.merge([data_mucape, data_mulcl, data_rh36mean, data_lr36]).load()
    
    
            data_selected = select_point_based_data(data, df, year).to_dataframe()
            
            
            all_data.append(data_selected)
            
        
        point_data = pd.concat(all_data)
    
        point_data.to_feather(os.path.join(CSV_DIR[:-10],'dialation_mucape/','1_env_variables_'+f)) # it saved as .csv extension but it should be .feather extension

    client.close()
    
        
if __name__ == "__main__":
    main()

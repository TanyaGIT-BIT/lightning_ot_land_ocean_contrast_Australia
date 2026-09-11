"""
Initial Parcel (Lifted Parcel Level) Parameters by Lightning/OT Category

Companion to pressure_level_profiles.py and screen_level_surface_conditions.py — same
categorized lightning/OT input (MUCAPE > 0 filtered) and region/warm-season filtering,
but extracts lifted-parcel-level temperature, pressure, and mixing ratio (with dew point
derived from mixing ratio), masked to MUCAPE > 0 conditions.

Purpose:
    Compute per-year, per-category (land/ocean, LS-only/LS+OT/OT-only) mean values and
    non-NaN counts for lifted parcel level temperature, pressure, and dew point, at
    lightning/OT detection points, for warm-season tropical Australia.

Input:
    - Categorized lightning/OT point CSVs (MUCAPE > 0 filtered):
      dialation_mucape/categorical_position/1_MUCAPE_0_*_3d.csv
    - BARRA-R2 hourly reanalysis: MULPLmixr, MULPLtemp, MULPLpres, MUCAPE
    - Region: tropical Australia (lat: -20 to -10, lon: 120 to 148)
    - Warm season: October-April, years 2016-2024
    - Requires local `atmos.moisture` module for dew point calculation

Output:
    Combined summary CSV (mean, non-NaN count per group/contrast/variable/year):
    /g/data/if69/tp4064/Project_1/categorical_position/BARRA_data_skew_T/
    warm_season_initial_parameters/1_initial_param.csv

Usage:
    Run: python initial_parcel_parameters.py
"""

# Standard libraries
import os
import glob

# Data manipulation
import pandas as pd
import numpy as np
import xarray as xr

# Parallel computing
import dask
import sys

sys.path.append('/g/data/if69/tp4064/old_before_new_data_merge/BARRA_combine_data')
from atmos import moisture

from dask.distributed import Client



def get_mucape_mask(year, DATA_DIR, warm_months):

    start = f"{year}-01-01"
    end   = f"{year}-12-31"

    ds = xr.open_mfdataset(
            get_files(
                os.path.join(DATA_DIR, 'MUCAPE', 'latest/'),
                start,
                end,
                warm_months=[10, 11, 12, 1, 2, 3, 4]
            ),
            chunks={},  # or {'time': 240} later
            combine='by_coords',
            preprocess=preprocess_fn )

    return ds['MUCAPE'] > 0

    

def yearly_data_processing(year, DATA_DIR, var,mask, warm_months=[10, 11, 12, 1, 2, 3, 4]):
    
    start = f"{year}-01-01"
    end   = f"{year}-12-31"
    
    df_all = xr.open_mfdataset(
            get_files(
                os.path.join(DATA_DIR, var, 'latest/'),
                start,
                end,
                warm_months=[10, 11, 12, 1, 2, 3, 4]
            ),
            chunks={},  # or {'time': 240} later
            combine='by_coords',
            preprocess=preprocess_fn
        )
   
    
    return df_all.where(mask)



# ── Helper: match monthly NetCDF files in a date range ────────────────────────
def get_files(path: str, start: str, end: str, warm_months) -> list[str]:
    months = pd.date_range(start=start, end=end, freq='MS')#.strftime('%Y%m')
    months = months[months.month.isin(warm_months)].strftime('%Y%m')
    files  = sorted(f for m in months for f in glob.glob(os.path.join(path, f'*-{m}.nc')))
    return files

# ──


def select_point_based_data(data,category,year, warm_months=(10, 11, 12, 1, 2, 3, 4)):
    """
    Select nearest grid-point data for OT/lightning points.
    """

    # Copy to avoid modifying original dataframe
    category = category.copy()

    # Convert time once
    category['time'] = pd.to_datetime(category['time'])

    # Single combined mask (faster + cleaner)
    mask = (
        (category['lat'].between(-20, -10)) &
        (category['lon'].between(120, 148)) &
        (category['time'].dt.year == year) &
        (category['time'].dt.month.isin(warm_months))
    )

    category = category.loc[mask]

    # Reduce precision slightly for cleaner nearest-neighbour matching
    category['lat'] = category['lat'].round(4)
    category['lon'] = category['lon'].round(4)

    # Convert directly to NumPy arrays (lighter than pandas Series)
    da_lat = xr.DataArray(category['lat'].to_numpy(), dims='points')
    da_lon = xr.DataArray(category['lon'].to_numpy(), dims='points')
    da_time = xr.DataArray(category['time'].to_numpy(), dims='points')

    # Nearest selection
    data_select = data.sel(
        lat=da_lat,
        lon=da_lon,
        time=da_time,
        method='nearest'
    )

    return data_select


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

    # Filter warm months (Oct–Apr) per file before concat
    # warm_mask = ds.time.dt.month.isin(warm_months)
    # ds = ds.isel(time=warm_mask)
    
    return ds


def main():
    global LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

    client = Client() #threads_per_worker=1, memory_limit = '16GB', n_workers = 4)
    
    
    var = ['MULPLmixr','MULPLtemp','MULPLpres']
    LAT_MIN, LAT_MAX = -20, -10
    LON_MIN, LON_MAX = 120, 148
    DATA_DIR   = '/g/data/ob53/BARRA2/output/reanalysis/AUST-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/'
    
    PAD = 2
    warm_months = [10, 11, 12, 1, 2, 3, 4]
    years = range(2016, 2025)
    CSV_DIR = "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/categorical_position/"
    
    
    csv_files = {
        "ls_only_land"   : "1_MUCAPE_0_ls_only_land_3d.csv",
        "ls_only_ocean"  : "1_MUCAPE_0_ls_only_ocean_3d.csv",
        "ls_ot_land"     : "1_MUCAPE_0_ls_ot_land_3d.csv",
        "ls_ot_ocean"    : "1_MUCAPE_0_ls_ot_ocean_3d.csv",
        "land_ot_no_ls"  : "1_MUCAPE_0_ot_only_land_3d.csv",
        "ocean_ot_no_ls" : "1_MUCAPE_0_ot_only_ocean_3d.csv",
    }
    
    data = {name: pd.read_csv(CSV_DIR + file) for name, file in csv_files.items()}
    
    CATEGORIES = {
        'OT only': (data['land_ot_no_ls'],  data['ocean_ot_no_ls']),
        'OT + LS': (data['ls_ot_land'],     data['ls_ot_ocean']),
        'LS only': (data['ls_only_land'],   data['ls_only_ocean']),
        }    
    
    dataset=[]
    
    
    for year in years:
    
        df = []
    
        print(year)
    
        mask = get_mucape_mask(year, DATA_DIR, warm_months)
    
        data_mulplmixr = yearly_data_processing(year, DATA_DIR, 'MULPLmixr', mask, warm_months)
        data_mulpltemp = yearly_data_processing(year, DATA_DIR, 'MULPLtemp', mask,  warm_months)
        data_mulplpress = yearly_data_processing(year, DATA_DIR, 'MULPLpres', mask,  warm_months)
        
        data_mulpldew = (
        moisture.dewpoint_temperature_from_mixing_ratio(
            data_mulplpress.MULPLpres,
            data_mulpltemp.MULPLtemp,
            data_mulplmixr.MULPLmixr
        ).to_dataset(name='MULPLdew'))
    
        data_all = xr.merge([data_mulpltemp, data_mulplpress,data_mulpldew], join = 'outer')
        data_all = data_all.load()
        
        rows = []
        for group, (land_df, ocean_df) in CATEGORIES.items():
            
            for contrast, cat_df in [('Land', land_df), ('Ocean', ocean_df)]:
                
                CHECK = select_point_based_data(data_all, cat_df,year, warm_months)
                
                mean_vals = CHECK.mean(dim='points', skipna=True).compute()
                nonnan_counts = CHECK.count(dim='points').compute()   # counts non-NaN values per variable
                
                
                for variable in CHECK.data_vars:
                    rows.append({
                        'group' : group,
                        'contrast': contrast,
                        'Variable' : variable,
                        'year': year,
                        'mean' : mean_vals[variable].item(),
                        'nonnan_count': nonnan_counts[variable].item(),
                    })
                
        
        summary = pd.DataFrame(rows)
        
        df.append(summary)
            
    
        dataset.append(pd.concat(df))
    
    pd.concat(dataset).to_csv(f'/g/data/if69/tp4064/Project_1/categorical_position/BARRA_data_skew_T/warm_season_initial_parameters/1_initial_param.csv')
       
        

    client.close()


if __name__ == '__main__':
    main()
    

"""
Pressure-Level Temperature/Dew Point Profiles by Lightning/OT Category

Purpose:
    For each categorized lightning/OT CSV (land/ocean, LS-only/LS+OT/OT-only, filtered to
    points where MUCAPE > 0), extract vertical profiles of temperature and dew point at
    10 pressure levels (1000-200 hPa) from BARRA-R2 reanalysis, sampled at the nearest
    grid point/time to each detection. Produces per-year, per-category mean profiles for
    building skew-T-style vertical structure comparisons.

Input:
    - Categorized lightning/OT point CSVs (region- and convective-condition-filtered):
      dialation_mucape/categorical_position/1_MUCAPE_0_*_3d.csv
      (Note: these are the same underlying points as the base categorized CSVs from
      06_categorical_lightning_OT_dilation.ipynb, further filtered to MUCAPE > 0.)
    - BARRA-R2 hourly reanalysis: ta<pressure> (temperature) and hus<pressure>
      (specific humidity) for pressure levels 1000, 950, 925, 850, 700, 600, 500,
      400, 300, 200 hPa
    - Region: Northern Australia (lat: -20 to -10, lon: 120 to 148)
    - Warm season: October-April, years 2016-2024
    - Requires local `atmos.thermo` module for dew point calculation

Processing:
    1. Load each categorized CSV, filter to the region and warm-season months.
    2. For each year and each pressure level, lazily load temperature and specific
       humidity fields (dask-delayed, month-by-month), sample at the nearest grid
       point/time to each detection, and compute dew point.
    3. Average temperature and dew point (and valid-data counts) across all detection
       points, per pressure level.
    4. Assemble a per-year vertical profile dataset (temperature, dew_point, counts by
       pressure) and save to CSV.
    5. A fresh Dask client is created per input CSV file; memory is explicitly cleaned
       up (gc.collect, client.run(gc.collect)) after each pressure level, year, and file
       to manage memory across this large batch job.

Output:
    Per-category, per-year profile CSVs saved to:
    /g/data/if69/tp4064/Project_1/categorical_position/BARRA_data_skew_T/
    warm_Season_mucape_0_pressure_level_param/1_<csv_stem>_<year>.csv

Usage:
    Run as a batch job via shell script (not interactive), e.g.:
"""

import gc
import dask
import dask.array
from dask.distributed import Client
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from datetime import datetime
import sys
import os
sys.path.append('/g/data/if69/tp4064/old_before_new_data_merge/BARRA_combine_data')
from atmos import thermo


# ── Constants ────────────────────────────────────────────────────────────────
BARRA_DIR = '/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr'
PRESSURE_LEVELS = ['1000', '950', '925', '850', '700', '600', '500', '400', '300', '200']
YEAR_RANGE = range(2016, 2025)

LAT_MAX, LAT_MIN = -10, -20
LON_MIN, LON_MAX = 120, 148


# ── Helper functions defined ONCE outside the loop ───────────────────────────
def select_warm_season(df):

    return df.loc[
        (df["lat"] >= LAT_MIN) &
        (df["lat"] <= LAT_MAX) &
        (df["lon"] >= LON_MIN) &
        (df["lon"] <= LON_MAX) &
        (df["time"].dt.month.isin([10, 11, 12, 1, 2, 3, 4]))
    ]
    
@dask.delayed
def open_barra_delayed(var, month, barra_dir=BARRA_DIR):
    file = sorted(glob(f'{barra_dir}/{var}/latest/{var}*{month.strftime("%Y%m")}.nc'))[0]
    return xr.open_dataset(file)


def open_barra_month(var, month, lats=646, lons=1082, dtype=np.float64):
    var_data = open_barra_delayed(var=var, month=month)[var].data
    times = pd.date_range(
        start=month,
        end=month + pd.tseries.offsets.MonthBegin(),
        freq='1h', inclusive='left'
    )
    return dask.array.from_delayed(var_data, (times.size, lats, lons), dtype)

    
def open_barra(var, var_renamed, months, barra_dir=BARRA_DIR, **kwargs):
    ref_file = sorted(glob(f'{barra_dir}/{var}/latest/{var}*{months[0].strftime("%Y%m")}.nc'))[0]
    ref = xr.open_dataset(ref_file)
    all_months = dask.array.concatenate(
        [open_barra_month(var=var, month=m, **kwargs) for m in months]
    )
    times = pd.date_range(
        start=months.min(),
        end=months.max() + pd.tseries.offsets.MonthBegin(),
        freq='1h', inclusive='left'
    )
    return xr.DataArray(
        all_months,
        dims=['time', 'lat', 'lon'],
        coords={
            'time': times,
            'lat': ref.lat,
            'lon': ref.lon,
            'pressure': ref.pressure,
            'crs': ref.crs,
        },
        name=var_renamed,
        attrs=ref[var].attrs
    ).chunk({'time': 100, 'lat': 100, 'lon': 100})

# CSV_FILES: active input set for this run.
# Other categorizations (base dilation set, MULCL-conditioned subsets) are commented
# out below — uncomment the relevant block to switch input sets for a different run.
#
# Note: ls_only_land_3d.csv (base set) and 1_MUCAPE_0_ls_only_land_3d.csv (active set)
# contain the same underlying points — the latter is further filtered to only points
# where MUCAPE > 0.
# "1_MULCL_greater_equal_1500m_mucape_0_ls_only_land_3d.csv" This has further condition for MULCL >= 1500m for the figure 6 of paper

CSV_FILES = [
    # "/g/data/if69/tp4064/Project_1/categorical_position/Dialation/ls_only_land_3d.csv",
    # "/g/data/if69/tp4064/Project_1/categorical_position/Dialation/ls_only_ocean_3d.csv",
    # "/g/data/if69/tp4064/Project_1/categorical_position/Dialation/ls_ot_land_3d.csv",
    # "/g/data/if69/tp4064/Project_1/categorical_position/Dialation/ls_ot_ocean_3d.csv",
    # "/g/data/if69/tp4064/Project_1/categorical_position/Dialation/ot_only_land_3d.csv",
    # "/g/data/if69/tp4064/Project_1/categorical_position/Dialation/ot_only_ocean_3d.csv",
    
    "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/categorical_position/1_MUCAPE_0_ls_only_land_3d.csv",
    "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/categorical_position/1_MUCAPE_0_ls_only_ocean_3d.csv",
    "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/categorical_position/1_MUCAPE_0_ls_ot_land_3d.csv",
    "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/categorical_position/1_MUCAPE_0_ls_ot_ocean_3d.csv",
    "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/categorical_position/1_MUCAPE_0_ot_only_land_3d.csv",
    "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/categorical_position/1_MUCAPE_0_ot_only_ocean_3d.csv",

    
    # "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/1_MULCL_greater_equal_1500m_mucape_0_ls_only_land_3d.csv",
    # "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/1_MULCL_less_1500m_mucape_0_ls_only_land_3d.csv",
    
    ]

def main():
    
    
    # your work here
    # ── Outer loop: one CSV file at a time ───────────────────────────────────────
    for csv_file in CSV_FILES:

        client = Client() # for each file 
        
        print(f"\n========== Processing file: {csv_file} ==========")
    
        # ── Load CSV ─────────────────────────────────────────────────────────────
        file_data = pd.read_csv(csv_file, parse_dates=["time"])
        file_data_warm = select_warm_season(file_data).copy()
        file_data_warm['time'] = pd.to_datetime(file_data_warm['time'])
        print('yes')
        # ── Main loop ────────────────────────────────────────────────────────────
        for year in YEAR_RANGE:
            print(f"\nProcessing year: {year}")
    
            months = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='1MS')
            print('yes')
            variable = file_data_warm[file_data_warm['time'].dt.year == year].copy()
            if variable.empty:
                print(f"  No data for {year}, skipping.")
                continue
    
            da_lon  = xr.DataArray(variable['lon'].values,  dims=['ls'])
            da_lat  = xr.DataArray(variable['lat'].values,  dims=['ls'])
            da_time = xr.DataArray(variable['time'].values, dims=['ls'])
    
           
    
            datasets_temp, datasets_dew = [], []
            counts_temp, counts_dew = [], []
    
            for pressure in PRESSURE_LEVELS:
                folder_ta  = f'ta{pressure}'
                folder_hus = f'hus{pressure}'
    
                data_ta  = open_barra(folder_ta,  'temperature',       months).drop_vars("crs")
                data_hus = open_barra(folder_hus, 'specific_humidity', months).drop_vars("crs")
    
                ta_sel  = data_ta.sel( lon=da_lon, lat=da_lat, time=da_time, method='nearest')
                hus_sel = data_hus.sel(lon=da_lon, lat=da_lat, time=da_time, method='nearest')
                ta_computed, hus_computed = dask.compute(ta_sel, hus_sel)
    
                dew_computed = thermo.dewpoint_temperature(
                    100 * ta_computed.pressure, ta_computed, hus_computed
                ).compute()
    
                mean_dims = [d for d in ta_computed.dims if d != 'pressure']
                datasets_temp.append(ta_computed.mean(dim=mean_dims))
                datasets_dew.append(dew_computed.mean(dim=mean_dims))
                counts_temp.append(ta_computed.count())
                counts_dew.append(dew_computed.count())
    
                data_ta.close();  data_hus.close()
                del data_ta, data_hus, ta_computed, hus_computed, dew_computed
                gc.collect()
                client.run(gc.collect)
    
    
            # ── Assemble and save ─────────────────────────────────────────────────
            final_dataset = xr.merge([
                xr.concat(datasets_temp, dim='pressure', combine_attrs='no_conflicts').to_dataset(name='temperature'),
                xr.concat(datasets_dew,  dim='pressure', combine_attrs='no_conflicts').to_dataset(name='dew_point'),
                xr.concat(counts_temp,   dim='pressure', combine_attrs='no_conflicts').to_dataset(name='count_temp'),
                xr.concat(counts_dew,    dim='pressure', combine_attrs='no_conflicts').to_dataset(name='count_dew'),
            ])
    
            # Use the csv filename stem to differentiate output files
            csv_stem = os.path.splitext(os.path.basename(csv_file))[0]
            # out_path= f"/g/data/if69/tp4064/Project_1/categorical_position/BARRA_data_skew_T/warm_Season_MULCL_condition/1_{csv_stem}_{year}.csv"
            out_path = f"/g/data/if69/tp4064/Project_1/categorical_position/BARRA_data_skew_T/warm_Season_mucape_0_pressure_level_param/1_{csv_stem}_{year}.csv"
            final_dataset.to_dataframe().to_csv(out_path, index=True)
            print(f"  Saved → {out_path}")
    
            del final_dataset, datasets_temp, datasets_dew, counts_temp, counts_dew
            gc.collect()
            client.run(gc.collect)
    
        # ── Cleanup after each file ───────────────────────────────────────────────
        del file_data, file_data_warm
        gc.collect()    

        try:
            client.close()
        except Exception as e:
            print(f"Warning: client.close() failed: {e}")

if __name__ == '__main__':
    main()

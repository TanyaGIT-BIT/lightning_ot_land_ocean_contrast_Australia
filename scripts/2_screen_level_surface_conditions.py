"""
Screen-Level (Surface) Temperature, Dew Point, and Pressure by Lightning/OT Category

Companion to pressure_level_profiles.py — same categorized lightning/OT input and
region/warm-season filtering, but extracts screen-level (surface) temperature, dew
point, and pressure instead of the 10-level upper-air profile. See
pressure_level_profiles.py for full documentation of the shared input data and
categorization logic.


Output:
    Per-category CSVs (one row per year) saved to:
    /g/data/if69/tp4064/Project_1/categorical_position/BARRA_data_skew_T/
    warm_Season_mucape_0_pressure_level_param/1_screen_level_mean_<csv_stem>.csv

Usage:
    Run: python screen_level_surface_conditions.py
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

# all this is only for warm season and only tropical Australia and for screen level diagnostic.

# ── Constants ────────────────────────────────────────────────────────────────
BARRA_DIR = '/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr'
pressure = 's' #screen level is 's'
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
            # 'pressure': ref.pressure,
            # 'crs': ref.crs,
        },
        name=var_renamed,
        attrs=ref[var].attrs
    ).chunk({'time': 100, 'lat': 100, 'lon': 100})


CSV_FILES = [
    # "/g/data/if69/tp4064/Project_1/categorical_position/Dialation/ls_only_land_3d.csv",
    # "/g/data/if69/tp4064/Project_1/categorical_position/Dialation/ls_only_ocean_3d.csv",
    # "/g/data/if69/tp4064/Project_1/categorical_position/Dialation/ls_ot_land_3d.csv",
    # "/g/data/if69/tp4064/Project_1/categorical_position/Dialation/ls_ot_ocean_3d.csv",
    # "/g/data/if69/tp4064/Project_1/categorical_position/Dialation/ot_only_land_3d.csv",
    # "/g/data/if69/tp4064/Project_1/categorical_position/Dialation/ot_only_ocean_3d.csv",

    "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/categorical_position/MUCAPE_0_ls_only_land_3d.csv",
    "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/categorical_position/MUCAPE_0_ls_only_ocean_3d.csv",
    "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/categorical_position/MUCAPE_0_ls_ot_land_3d.csv",
    "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/categorical_position/MUCAPE_0_ls_ot_ocean_3d.csv",
    "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/categorical_position/MUCAPE_0_ot_only_land_3d.csv",
    "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/categorical_position/MUCAPE_0_ot_only_ocean_3d.csv",
    
    # "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/1_MULCL_greater_equal_1500m_mucape_0_ls_only_land_3d.csv",
    # "/g/data/if69/tp4064/Project_1/categorical_position/dialation_mucape/1_MULCL_less_1500m_mucape_0_ls_only_land_3d.csv",
]

def main():
    client = Client()

    # ── Outer loop: one CSV file at a time ───────────────────────────────────────
    for csv_file in CSV_FILES:
        print(f"\n========== Processing file: {csv_file} ==========")
    
        # ── Load CSV ─────────────────────────────────────────────────────────────
        file_data = pd.read_csv(csv_file, parse_dates=["time"])
        file_data_selected = select_warm_season(file_data).copy()
        file_data_selected['time'] = pd.to_datetime(file_data_selected['time'])
        # print('yes')
    
        # ── Initialise ────────────────────────────────────────────────────────────
        datasets_temp, datasets_dew, datasets_press = [], [], []
        counts_temp, counts_dew, counts_press = [], [], []
        years_list = []
        
        # ── Main loop ────────────────────────────────────────────────────────────
    
        for year in YEAR_RANGE:
            print(f"\nProcessing year: {year}")
    
            months = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='1MS')
            # print('yes')
            variable = file_data_selected[file_data_selected['time'].dt.year == year].copy()
            if variable.empty:
                print(f"  No data for {year}, skipping.")
                continue
    
            da_lon  = xr.DataArray(variable['lon'].values,  dims=['ls'])
            da_lat  = xr.DataArray(variable['lat'].values,  dims=['ls'])
            da_time = xr.DataArray(variable['time'].values, dims=['ls'])
    
            
            # print(pressure)
            folder_ta  = f'ta{pressure}'
            folder_hus = f'hus{pressure}'
            folder_press = 'ps'
    
            data_ta  = open_barra(folder_ta,  'temperature',       months)#.drop_vars(["crs","height"])
            data_hus = open_barra(folder_hus,'specific_humidity',months)#.
            data_press = open_barra(folder_press, 'pressure', months)#.drop_vars("crs")
            
    
            ta_sel  = data_ta.sel( lon=da_lon, lat=da_lat, time=da_time, method='nearest')
            hus_sel = data_hus.sel(lon=da_lon, lat=da_lat, time=da_time, method='nearest')
            ps_sel = data_press.sel(lon=da_lon, lat=da_lat, time=da_time, method='nearest')
            
            ta_computed, hus_computed, ps_computed = dask.compute(ta_sel, hus_sel, ps_sel )
    
            dew_computed = thermo.dewpoint_temperature(ps_computed, ta_computed, hus_computed).compute()
    
            ps_computed = ps_computed/100 # converting to hpa, to align with other pressure levels
    
            mean_dims = [d for d in ta_computed.dims if d != 'pressure']
            datasets_temp.append(ta_computed.mean(dim=mean_dims))
            datasets_dew.append(dew_computed.mean(dim=mean_dims))
            datasets_press.append(ps_computed.mean(dim = mean_dims))
            counts_temp.append(ta_computed.count())
            counts_dew.append(dew_computed.count())
            counts_press.append(ps_computed.count())
            years_list.append(year)
    
            data_ta.close();  data_hus.close(); data_press.close()
            del data_ta, data_hus, ta_computed, hus_computed, dew_computed, ps_computed
            gc.collect()
            client.run(gc.collect)
    
    
            # ── Assemble and save ─────────────────────────────────────────────────
    
        final_dataset = xr.Dataset(
            { 
                'surface_temp' : (['year'], datasets_temp),
                'surface_dew' : (['year'], datasets_dew),
                'surface_press' : (['year'], datasets_press),
                'count_temp' : (['year'], counts_temp),
                'count_dew' : (['year'], counts_dew),
                'Count_press' : (['year'], counts_press),
            },
            coords = {'year':years_list}
        )
    
        # # Use the csv filename stem to differentiate output files, save based on input file type
        csv_stem = os.path.splitext(os.path.basename(csv_file))[0]
        # out_path =f"/g/data/if69/tp4064/Project_1/categorical_position/BARRA_data_skew_T/warm_season_screen_parameters/screen_level_mean_{csv_stem}.csv"
        # out_path =f"/g/data/if69/tp4064/Project_1/categorical_position/BARRA_data_skew_T/warm_Season_MULCL_condition/1_screen_level_mean_{csv_stem}.csv"
        out_path = f"/g/data/if69/tp4064/Project_1/categorical_position/BARRA_data_skew_T/warm_Season_mucape_0_pressure_level_param/1_screen_level_mean_{csv_stem}.csv"
        final_dataset.to_dataframe().to_csv(out_path, index=True)
        # print(f"  Saved → {out_path}")
    
        del final_dataset, datasets_temp, datasets_dew, counts_temp, counts_dew
        gc.collect()
        client.run(gc.collect)
    
        # ── Cleanup after each file ───────────────────────────────────────────────
        del file_data, file_data_selected
        gc.collect()
            
    client.close()

if __name__ == '__main__':
    main()




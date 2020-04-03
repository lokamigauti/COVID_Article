"""
Script to extract meteorological data from ERA5 reanalysis.
"""
import xarray as xr
from pathlib import Path


def read_data(path_to_files, vars_filenames):
    """
    Method to read and preprocess the ncdf files
    Parameters
    ----------
    path_to_files Path
    vars_filenames Dict

    Returns xr.DataSet
    -------

    """
    list_of_arrays = []
    years = [2020]
    for var in vars_filenames.keys():
        print(f'Reading variable {var}')
        list_of_years = []
        for year in years:
            print(f'Reading year {year}')
            filepath = path_to_files / vars_filenames[var].format(year=year)
            array = xr.open_dataarray(filepath)
            array = array.chunk({'latitude': 10, 'longitude': 10})
            array = array.assign_coords(longitude=(array.coords['longitude'].values + 180) % 360 - 180)
            array = array.sortby('longitude')
            array = array.sortby('latitude')
            list_of_years.append(array)
        array_concat_years = xr.concat(list_of_years, dim='time')
        list_of_arrays.append(array_concat_years)
    dataset = xr.merge(list_of_arrays)
    return dataset


if __name__ == '__main__':

    vars_filenames = {  # Dictionary mapping variables and their filenames
        'uv': 'UV_sfc_ERA5_6hr_{year}010100-{year}123118.nc',
        'tmp': 'tmp_2m_ERA5_6hr_{year}010100-{year}123118.nc',
    }
    locations = {
        'new_york': {
            'latitude': 40.7,
            'longitude': -73.9
        },
        'philadelphia': {
            'latitude': 39.95,
            'longitude': -75.13
        },
        'daegu': {
            'latitude': 35.84,
            'longitude': 128.57,
        }
    }
    path_to_files = Path('~/phd/data/ERA5/')
    dataset = read_data(path_to_files, vars_filenames)
    # locations_array = xr.zeros_like(dataset['uvb']).isel(time=0, expver=0).drop('time').drop('expver')
    # locations_array.where()
    #
    datasets_by_location = []
    for location in locations.keys():
        coords = locations[location]
        d = dataset.sel(coords, method='nearest')
        d1 = d.where(d.expver==1, drop=True).squeeze('expver').drop('expver')
        d5 = d.where(d.expver == 5, drop=True).squeeze('expver').drop('expver')
        d = xr.concat([d1.dropna('time'), d5.dropna('time')], dim='time')
        datasets_by_location.append(d.assign_coords(location_name=location))

    datasets_by_location = xr.concat(datasets_by_location, 'location_name')
    datasets_by_location.to_dataframe().to_csv('data/meteorological_variables.csv')
    datasets_by_location.resample(time='1D').mean().to_dataframe().to_csv('data/meteorological_variables_daily_mean.csv')


    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    datasets_by_location['uvb'].sel(expver=1).plot.line(x='time')
    plt.show()
    datasets_by_location['uvb'].sel(expver=1).resample(time='1D').mean().plot.line(x='time')

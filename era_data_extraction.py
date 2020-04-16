"""
Script to extract meteorological data from ERA5 reanalysis.
"""
import xarray as xr
from pathlib import Path
import pandas as pd
import urllib.request, json
import numpy as np
import locale
def read_era(path_to_files, vars_filenames):
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



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def read_aqi():
    path_to_aqi = 'data/waqi-covid19-airqualitydata.csv'
    aqi = pd.read_csv(path_to_aqi)
    aqi = aqi.loc[aqi['Specie'] == 'pm25']
    aqi = aqi.loc[aqi['Country'] == 'US']

    with urllib.request.urlopen("https://aqicn.org/data-platform/covid19/airquality-covid19-cities.json") as url:
        data = json.loads(url.read().decode())

    aqi_metadata = {}
    for l in data['data']:
        aqi_metadata[l['Place']['name']] = l['Place']['geo']
    aqi['latitude'] = np.nan
    aqi['longitude'] = np.nan
    for city in aqi_metadata.keys():
        aqi['latitude'].where(aqi['City'] != city, other=aqi_metadata[city][0], inplace=True)
        aqi['longitude'].where(aqi['City'] != city, other=aqi_metadata[city][1], inplace=True)
    return aqi, aqi_metadata

def read_JHopkins():
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
    data = pd.read_csv(url)
    data = data.drop(['UID', 'iso2', 'iso3', 'FIPS', 'Combined_Key', 'Country_Region', 'code3'], axis=1)
    data = data.melt(id_vars=['Province_State', 'Admin2', 'Lat', 'Long_'], value_name='cases', var_name='time')
    data = data.set_index(['Province_State', 'Admin2', 'time']).to_xarray()
    data = data.assign_coords(time=[pd.Timestamp(x) for x in data.time.values])
    return data

if __name__ == '__main__':

    vars_filenames = {  # Dictionary mapping variables and their filenames
        'uv': 'UV_sfc_ERA5_6hr_{year}010100-{year}123118.nc',
        'tmp': 'tmp_2m_ERA5_6hr_{year}010100-{year}123118.nc',
        'direct_solar_radiation': 'direct_solar_radiation_sfc_ERA5_6hr_{year}010100-{year}123118.nc'
    }

    quarentine = pd.read_csv('data/quarentine_USA.csv', sep=';')
    path_to_files = Path('~/phd/data/ERA5/')
    dataset = read_era(path_to_files, vars_filenames)
    aqi, aqi_metadata = read_aqi()
    jHopkins = read_JHopkins()
    dist_threshold = 1
    datasets_by_location = []
    states = jHopkins.Province_State.values
    locations = jHopkins.Admin2.values


    for state in states:
        try:
            quarentine_date = quarentine.loc[quarentine['State'].str.contains(state)]['Effective Stay Home'].values[0]
            quarentine_date = quarentine_date + '/2020'
            day, month, year = quarentine_date.split('/')
            month = month.capitalize()
            month = 'Apr' if month == 'Abr' else month
            quarentine_date = pd.to_datetime(day + '/' + month + '/' + year, format='%d/%b/%Y')
        except:
            quarentine_date = pd.to_datetime('01/Jan/2090', format='%d/%b/%Y' )
        for location in locations:

            if location in aqi['City'].unique():

                print(f'Great, {location} exists in both datasets')
                lat, lon = aqi_metadata[location]

                jH = jHopkins.sel(Province_State=state).where(jHopkins.Admin2 == location, drop=True)
                jH = jH.dropna('time', how='all')
                if jH.time.values.shape[0] > 0:
                    jHlat = jH.isel(time=0, Admin2=0).Lat.values
                    jHlon = jH.isel(time=0, Admin2=0).Long_.values
                    dist = ((jHlat - lat) ** 2 + (jHlon - lon) ** 2) ** 0.5
                    jH = jH.drop('Province_State')

                    d = dataset.sel(latitude=lat, longitude=lon, method='nearest')
                    if dist.min() < dist_threshold:
                        jH = jH['cases']
                        d1 = d.where(d.expver == 1, drop=True).squeeze('expver').drop('expver')
                        d5 = d.where(d.expver == 5, drop=True).squeeze('expver').drop('expver')
                        d = xr.concat([d1.dropna('time'), d5.dropna('time')], dim='time')
                        d = d.sortby('time')
                        d = d.resample(time='1D').mean()

                        aqi_temp = aqi.loc[aqi['City'] == location][['median', 'Date']].set_index('Date').to_xarray().rename(
                            {'Date': 'time'})
                        aqi_temp = aqi_temp.assign_coords(time=[pd.Timestamp(x) for x in aqi_temp.time.values])
                        aqi_temp = aqi_temp.expand_dims(['latitude', 'longitude']).assign_coords(latitude=[lat], longitude=[lon])
                        jH = jH.expand_dims(['latitude', 'longitude']).assign_coords(latitude=[lat], longitude=[lon])
                        jH.name = 'covid_cases'

                        jH_diff = jH.differentiate('time', datetime_unit='D').where(jH > 5, 0)

                        jH_diff.name = 'covid_cases_first_derivative'
                        jH_diff2 = jH_diff.differentiate('time', datetime_unit='D').where(jH > 5, 0)
                        jH_diff2.name = 'covid_cases_second_derivative'
                        jH = jH.to_dataset()
                        jH = jH.isel(Admin2=0).drop('Admin2')
                        jH_diff = jH_diff.to_dataset()
                        jH_diff = jH_diff.isel(Admin2=0).drop('Admin2')
                        jH_diff2 = jH_diff2.to_dataset()
                        jH_diff2 = jH_diff2.isel(Admin2=0).drop('Admin2')
                        try:
                            d = xr.merge([d, aqi_temp, jH, jH_diff, jH_diff2])
                            d = d.rename({'median': 'median_mp25'}).isel(latitude=0, longitude=0)
                            d = d.assign_coords(location_name=location)
                            d['is_quarentined'] = ('time'), [True if x > quarentine_date else False for x in
                                                             d.time.values]
                            datasets_by_location.append(d)
                        except:
                            print('Could not merge datasets')
                            pass
                    else:
                        print('Too bad, aqi and jH points are too far off.')
                else:
                    print(f'Too bad, jHopkins data for {location} is empty')
        else:
            print(f'Too bad, {location} is not available in the aqi dataset.')

    datasets_by_location = xr.concat(datasets_by_location, dim='location_name')
    # datasets_by_location.to_dataframe().to_csv('data/meteorological_variables.csv')
    datasets_by_location.to_dataframe().to_csv('data/meteorological_variables_daily_mean.csv')
    # datasets_by_location.resample(time='1D').mean().to_netcdf('data/dataset.nc')
    #
    # aqi['median']
    #
    #
    #
    # df.loc[df['column_name'] == some_value]
    # aqi = aqi.set_index(['Date', 'Country', 'City'])
    # aqi.to_xarray()
    # import matplotlib.pyplot as plt
    # plt.style.use('seaborn')
    # datasets_by_location['uvb'].sel(expver=1).plot.line(x='time')
    # plt.show()
    # datasets_by_location['uvb'].sel(expver=1).resample(time='1D').mean().plot.line(x='time')

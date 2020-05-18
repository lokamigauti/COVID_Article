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
    aqi = aqi.loc[aqi['Country'] == 'US']
    aqi25 = aqi.loc[aqi['Specie'] == 'pm25']
    aqi10 = aqi.loc[aqi['Specie'] == 'pm10']
    aqi = pd.concat([aqi25, aqi10], axis=0)
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
    cities = pd.read_csv('data/uscities.csv')

    quarentine = pd.read_csv('data/quarentine_USA.csv', sep=';')
    path_to_files = Path('~/phd/data/ERA5/')
    dataset = read_era(path_to_files, vars_filenames)
    aqi, aqi_metadata = read_aqi()
    jHopkins = read_JHopkins()
    dist_threshold = 1
    datasets_by_location = []
    states = jHopkins.Province_State.values
    aqi_cities = aqi['City'].unique()

    for state in states:
        print(f'*----- Doing state {state} -----*')
        try:
            quarentine_date = quarentine.loc[quarentine['State'].str.contains(state)]['Effective Stay Home'].values[0]
            quarentine_date = quarentine_date + '/2020'
            day, month, year = quarentine_date.split('/')
            month = month.capitalize()
            month = 'Apr' if month == 'Abr' else month
            quarentine_date = pd.to_datetime(day + '/' + month + '/' + year, format='%d/%b/%Y')
        except:
            quarentine_date = pd.to_datetime('01/Jan/2090', format='%d/%b/%Y' )
        counties_in_state = cities.loc[cities['state_name'] == state]['county_name'].unique()
        print(counties_in_state)

        for county in counties_in_state:
            cities_in_state = cities.loc[cities['state_name'] == state]

            cities_in_county = cities_in_state.loc[cities_in_state['county_name'] == county]
            # print('There are {} cities in county {}'.format(str(cities_in_county.shape[0]), county))
            jH = jHopkins.sel(Province_State=state).where(jHopkins.Admin2 == county, drop=True)
            jH = jH.dropna('time', how='all')
            if jH.time.values.shape[0] == 0:
                print(f'County {county} is empty in the time axis.')
                break

            if jH.Admin2.values.shape[0] == 0:
                print(f'State {state} has only one conty: {county}')
                jHlat = jH.isel(time=0).Lat.values
                jHlon = jH.isel(time=0).Long_.values
            else:
                jHlat = jH.isel(time=0, Admin2=0).Lat.values
                jHlon = jH.isel(time=0, Admin2=0).Long_.values
            city = None
            print('Cities in county are: ')
            print(cities_in_county['city'].values)
            for _city in cities_in_county['city'].values:

                if _city in aqi_cities:
                    print(f'City {_city} matches.')

                    aqilat, aqilon = aqi_metadata[_city]
                    dist = ((jHlat - aqilat) ** 2 + (jHlon - aqilon) ** 2) ** 0.5
                    if dist > dist_threshold:
                        print(f'{_city} is too far, probably belongs to a different state.')
                    else:
                        city = _city

            if isinstance(city, type(None)):
                print(f'There is no aqi data for county {county}')
            else:
                d = dataset.sel(latitude=aqilat, longitude=aqilon, method='nearest')
                jH = jH['cases']
                d1 = d.where(d.expver == 1, drop=True).squeeze('expver').drop('expver')
                d5 = d.where(d.expver == 5, drop=True).squeeze('expver').drop('expver')
                d = xr.concat([d1.dropna('time'), d5.dropna('time')], dim='time')
                d = d.sortby('time')
                d = d.resample(time='1D').mean()


                aqi_temp = aqi.loc[aqi['City'] == city][['Specie', 'median', 'Date']].set_index(
                    'Date').to_xarray().rename(
                    {'Date': 'time'})
                aqi_temp = aqi_temp.assign_coords(time=[pd.Timestamp(x) for x in aqi_temp.time.values])
                aqi_temp = aqi_temp.sortby('time')
                aqi_temp = xr.merge([
                    aqi_temp.where(aqi_temp.Specie == 'pm10', drop=True)['median'].rename('median_pm10'),
                    aqi_temp.where(aqi_temp.Specie == 'pm25', drop=True)['median'].rename('median_pm25'),
                ])

                aqi_temp = aqi_temp.expand_dims(['latitude', 'longitude']).assign_coords(
                    latitude=[aqilat],
                    longitude=[aqilon],

                )
                jH = jH.expand_dims(['latitude', 'longitude']).assign_coords(latitude=[aqilat], longitude=[aqilon])
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
                    d = d.assign_coords(location_name=city)
                    d = d.assign_coords(population=cities_in_county.loc[cities_in_county['city']==city]['population'].values)
                    d = d.assign_coords(density=cities_in_county.loc[cities_in_county['city']==city]['density'].values)
                    d = d.isel(latitude=0, longitude=0, population=0, density=0)

                    d['is_quarentined'] = ('time'), [True if x > quarentine_date else False for x in
                                                     d.time.values]
                    datasets_by_location.append(d)
                except:
                    print('Could not merge datasets')
                    pass


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

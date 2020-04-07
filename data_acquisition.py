import pandas as pd
import io
import requests
url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
s=requests.get(url).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))

c = c.melt(id_vars=['Province/State', 'Country/Region','Lat','Long'], var_name='Date', value_name='Infected')
c['Date'] = pd.to_datetime(c['Date'])
c = c.set_index(['Lat','Long', 'Date'])
c = c.where(c['Country/Region']=='Brasil')
da = c.to_xarray()
da.isel(Date=0).plot()

path_to_aqi = 'data/waqi-covid19-airqualitydata.csv'
pd.read_csv(path_to_aqi)
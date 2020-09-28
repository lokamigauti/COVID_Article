"""
Analysis of granger causality between pollutants and covid.
Authors: Gabriel Perez and Leo Kamigauti
"""

import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy
import cartopy.feature as cfeature
import cmasher as cmr
from GrangerCausality.tools.causality import GrangerCausality, NotEnoughContinuousData, NotEnoughData
import matplotlib


# ---- Cities to remove based on quality control ---- #
rm_pm25 = ['Columbia', 'Richmond', 'San Diego']
rm_pm10 = [
    'Austin', 'Boston', 'Brooklyn', 'Charlotte', 'Columbia', 'Columbus', 'Dallas', 'Fort Worth', 'Houston', 'Jackson',
    'Little Rock', 'Manhattan', 'Memphis', 'Nashville', 'Oakland', 'Philadelphia', 'Portland',
    'Providence', 'Queens', 'Raleigh', 'Richmond', 'Sacramento', 'Salem', 'Salt Lake City', 'San Antonio',
    'San Diego', 'San Francisco', 'Seattle', 'Springfield', 'Staten Island', 'Tallahassee'
]
rm_no2 = [
            'Austin', 'Charlotte', 'Columbia', 'Dallas', 'Detroit', 'Fort Worth', 'Honolulu', 'Houston',
            'Jackson', 'Little Rock', 'Madison', 'Memphis', 'Miami', 'Nashville', 'Omaha', 'Salem', 'San Antonio',
            'San Diego', 'Springfield', 'Tallahassee'
]
rm_co = [
    'Austin', 'Baltimore', 'Charlotte', 'Columbia', 'Columbus', 'Dallas', 'Detroit', 'El Paso', 'Fort Worth',
    'Honolulu', 'Houston', 'Las Vegas', 'Little Rock', 'Madison', 'Memphis', 'Nashville', 'Oklahoma City',
    'Philadelphia', 'Richmond', 'Sacramento', 'Salem', 'San Antonio', 'San Diego', 'Springfield'
]

# ---- Reading relevant variables from dataset ---- #
variables = ['covid_cases',
             'covid_cases_first_derivative',
             'covid_cases_second_derivative',
             'covid_cases_first_derivative_smooth',
             'covid_cases_second_derivative_smooth']
filepath = 'data/meteorological_variables_daily_mean.csv'
df = pd.read_csv(filepath)
ds = df.set_index(['time', 'location_name'])
ds = ds.to_xarray()
ds = ds.assign_coords(time=pd.to_datetime(ds.time.values))
ds_pm25 = ds['median_pm25']
ds_pm10 = ds['median_pm10']
ds_no2 = ds['median_no2']
ds_co = ds['median_co']
ds_pm25['exclude_PM25'] = 'location_name', [True if x in rm_pm25 else False for x in ds_pm25.location_name.values]
ds_pm10['exclude_PM10'] = 'location_name', [True if x in rm_pm10 else False for x in ds_pm10.location_name.values]
ds_no2['exclude_NO2'] = 'location_name', [True if x in rm_no2 else False for x in ds_no2.location_name.values]
ds_co['exclude_CO'] = 'location_name', [True if x in rm_co else False for x in ds_co.location_name.values]
predictors = xr.concat([ds_no2, ds_pm25, ds_pm10, ds_co], dim=pd.Index(['NO2', 'PM25', 'PM10', 'CO'], name='Variable'))
ds_total = ds.copy()
predictors.name = 'predictors'
ds = xr.merge([ds[variables], predictors])
ds = ds.where(ds.covid_cases > 5)  # Analysis starts when there is at least 5 cases in a city
ds = ds.stack({'samples': ['location_name', 'time']}).dropna('samples', how='all').unstack()
ds = ds.sortby('time')

# --- Loop through predictors CO, NO2, PM25 and PM10 and evaluate granger causality against cases second derivative
predictor_list = []
for predictor in ds.predictors.Variable.values:
    ds_temp = ds.predictors.sel(Variable=predictor)
    ds_temp = ds_temp.where(ds_temp['exclude_' + predictor] == 0, drop=True)
    location_list = []
    failed_locations = 0
    for location in ds_temp.location_name.values:
        da_x = ds_temp.sel(location_name=location).copy()
        da_x = da_x.expand_dims('Variable')
        da_y = ds.sel(location_name=location).covid_cases_second_derivative.copy()

        #  preprocess: removing long term trends to make ts stationary
        da_y_original = da_y.copy()

        da_x = da_x / np.abs(da_x).max('time')
        da_x = da_x - da_x.mean('time')
        da_y = da_y - da_y.mean('time')
        da_y = da_y / np.abs(da_y.max('time'))
        da_x = da_x.sortby('time')
        da_y = da_y.sortby('time')
        gc = GrangerCausality(sampledim='time', featuredim='Variable')
        try:
            pvalue_array = gc.run(da_x, da_y, detrend=True, granger_test='ssr_ftest', testtype='ct',
                                  test_stationarity=True, maxlag=17, critical_level='5%')
        except (ValueError, NotEnoughContinuousData, NotEnoughData):
            failed_locations += 1
        else:
            pvalue_array['location_name'] = location
            pvalue_array = pvalue_array.expand_dims('location_name')
            location_list.append(pvalue_array)
    print(f'{failed_locations} failed locations for predictor {predictor}.')
    predictor_list.append(xr.concat(location_list, dim='location_name'))

da_granger = xr.concat(predictor_list, dim='Variable')
da_granger = da_granger.assign_coords(State=ds_total.Province_State.isel(time=-1).drop('time'))
da_granger = da_granger.assign_coords(Density=ds_total.density.isel(time=0).drop('time'))


da_granger = da_granger.where(~xr.ufuncs.isnan(da_granger), 999)
minimal_lag = da_granger.argmin('lags', skipna=True) + 1  # plus one because argmin is zero based and lags start on one
da_granger = da_granger.where(da_granger < 999)

#
# # ==== Plot lagged optmium granger
# plt.style.use('bmh')
# fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
# locs_dict ={'PM25': ['Oklahoma City', 'Atlanta', 'Jacksonville', 'Phoenix'],
#        'PM10': ['Oklahoma City', 'Atlanta', 'Jacksonville', 'Indianapolis'],
#             'median_no2': ['Hartford', 'Chicago', 'Detroit', 'Las Vegas'],
#
#             }
#
# var = 'PM25'
# locs = locs_dict[var]
#
# for id, loc in enumerate(locs):
#     shift = da_granger.sel(location_name=loc, Variable=var).argmax().values + 1
#     ax=axs.flatten()[id]
#     pm_toplot = ds.sel(Variable=var, location_name=loc).shift(time=shift)['predictors']
#     pm_toplot = pm_toplot - pm_toplot.mean('time')
#     pm_toplot = pm_toplot/np.max(np.abs(pm_toplot))
#
#     cases_toplot = ds.sel(location_name=loc)['cases_second_derivative']
#     cases_toplot = cases_toplot - pm_toplot.mean('time')
#     cases_toplot = cases_toplot/np.max(np.abs(cases_toplot))
#     xticks = []
#     xticklabels = []
#     for idx, time in enumerate(pm_toplot.dropna('time').time.values):
#         if idx % 30 == 0:
#             xticks.append(time)
#             xticklabels.append(pd.Timestamp(time).strftime('%m/%d'))
#     ax.plot(cases_toplot.time.values, cases_toplot.values)
#     ax.plot(pm_toplot.time.values, pm_toplot.values)
#     # ax.scatter(pm_toplot.values, cases_toplot.values)
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xticklabels, rotation=40, ha='right')
#     ax.set_title(f'a) {loc}. Lag = {int(shift)} days.', loc='left', size='small')
# fig.legend(['PM25', 'Cases second deriv.'], loc='lower left')
# plt.show()
#

# ---- Assembling data tables and exporting to Latex ---- #
minimal_lag.name = 'Lag'
pvalue = da_granger.min('lags')
pvalue.name = 'p-value'

da_quants = ds.quantile(dim='time', q=[0.1, 0.9])
dsss = xr.merge([minimal_lag, pvalue])
df_quants = da_quants['predictors'].to_dataframe()
df_quants = df_quants.pivot_table(columns=['quantile', 'Variable'])

dsss = dsss.rename(location_name='City')

df1 = dsss.isel(Variable=0).drop('Variable').sortby('p-value').to_dataframe()
df1_aux = da_quants['predictors'].isel(Variable=0).drop('Variable').rename(location_name='City').to_dataframe().unstack(0)
df1 = pd.concat([df1, df1_aux], axis=1)
df1 = df1.rename(columns={('predictors', 0.1): '10% quantile'})
df1 = df1.rename(columns={('predictors', 0.9): '90% quantile'})
df1 = df1.dropna(axis=0)
df1 = df1.reset_index().rename({'index': 'City'}, axis=1)

df2 = dsss.isel(Variable=1).drop('Variable').sortby('p-value').to_dataframe()
df2_aux = da_quants['predictors'].isel(Variable=1).drop('Variable').rename(location_name='City').to_dataframe().unstack(0)
df2 = pd.concat([df2, df2_aux], axis=1)
df2 = df2.rename(columns={('predictors', 0.1): '10% quantile'})
df2 = df2.rename(columns={('predictors', 0.9): '90% quantile'})
df2 = df2.dropna(axis=0)
df2 = df2.reset_index().rename({'index': 'City'}, axis=1)

df3 = dsss.isel(Variable=2).drop('Variable').sortby('p-value').to_dataframe()
df3_aux = da_quants['predictors'].isel(Variable=2).drop('Variable').rename(location_name='City').to_dataframe().unstack(0)
df3 = pd.concat([df3, df3_aux], axis=1)
df3 = df3.rename(columns={('predictors', 0.1): '10% quantile'})
df3 = df3.rename(columns={('predictors', 0.9): '90% quantile'})
df3 = df3.dropna(axis=0)
df3 = df3.reset_index().rename({'index': 'City'}, axis=1)


df4 = dsss.isel(Variable=3).drop('Variable').sortby('p-value').to_dataframe()
df4_aux = da_quants['predictors'].isel(Variable=3).drop('Variable').rename(location_name='City').to_dataframe().unstack(0)
df4 = pd.concat([df4, df4_aux], axis=1)
df4 = df4.rename(columns={('predictors', 0.1): '10% quantile'})
df4 = df4.rename(columns={('predictors', 0.9): '90% quantile'})
df4 = df4.dropna(axis=0)
df4 = df4.reset_index().rename({'index': 'City'}, axis=1)

df1 = df1.round(decimals=3)
df2 = df2.round(decimals=3)
df3 = df3.round(decimals=3)
df4 = df4.round(decimals=3)
vars = dsss.Variable.values
df1.to_latex(f'data/granger_table_{vars[0]}.tex', index=False)
df2.to_latex(f'data/granger_table_{vars[1]}.tex', index=False)
df3.to_latex(f'data/granger_table_{vars[2]}.tex', index=False)
df4.to_latex(f'data/granger_table_{vars[3]}.tex', index=False)
df1.to_csv(f'data/granger_table_{vars[0]}.csv', index=False)
df2.to_csv(f'data/granger_table_{vars[1]}.csv', index=False)
df3.to_csv(f'data/granger_table_{vars[2]}.csv', index=False)
df4.to_csv(f'data/granger_table_{vars[3]}.csv', index=False)





# ---- Plotting ---- #

plt.style.use('bmh')
fig, axs = plt.subplots(2, 4, figsize=[8, 4])
df1.plot.scatter(ax=axs[0, 0], x='10% quantile', y='p-value')
df1.plot.scatter(ax=axs[1, 0], x='90% quantile', y='p-value')
axs[0, 0].set_title(f'a) {vars[0]} ', loc='left')
axs[1, 0].set_title(f'b) {vars[0]} ', loc='left')
df2.plot.scatter(ax=axs[0, 1], x='10% quantile', y='p-value')
df2.plot.scatter(ax=axs[1, 1], x='90% quantile', y='p-value')
axs[0, 1].set_title(rf'c) {vars[1]}', loc='left')
axs[1, 1].set_title(f'd) {vars[1]}', loc='left')

df3.plot.scatter(ax=axs[0, 2], x='10% quantile', y='p-value')
df3.plot.scatter(ax=axs[1, 2], x='90% quantile', y='p-value')
axs[0, 2].set_title(rf'e) {vars[2]} ', loc='left')
axs[1, 2].set_title(rf'f) {vars[2]} ', loc='left')
df4.plot.scatter(ax=axs[0, 3], x='10% quantile', y='p-value')
df4.plot.scatter(ax=axs[1, 3], x='90% quantile', y='p-value')
axs[0, 3].set_title(f'g) {vars[3]}', loc='left')
axs[1, 3].set_title(f'h) {vars[3]}', loc='left')
plt.show()



dfs = {vars[0]: df1, vars[1]: df2, vars[2]: df3, vars[3]: df4}


variables = ['PM25', 'PM10', 'NO2', 'CO']
subtitles = ['a', 'b', 'c', 'd']
xs = ds_total.sel(location_name=da_granger.location_name).isel(time=0).longitude
ys = ds_total.sel(location_name=da_granger.location_name).isel(time=0).latitude
avg_pol = ds_total.sel(location_name=da_granger.location_name).median_pm25.mean('time')
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
font_out = {'size': 5}
font_in = {'size': 4}

matplotlib.rc('font', **font_out)
fig, axs = plt.subplots(4, 1, subplot_kw={'projection': ccrs.Orthographic(-100, 35)})
for idx, var in enumerate(variables):
    variable = variables[idx]
    ax = axs.flatten()[idx]
    ax.coastlines(linewidth=.25)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=.5)
    ax.add_feature(states_provinces, edgecolor='gray', linewidth=.25)
    # ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.gridlines(linewidth=.25)
    ax.set_title(f'{subtitles[idx]}) {variable}', loc='left')
    # ax.set_xlim([-130, -80])

    p = ax.scatter(x=xs.values, y=ys.values, s=4, #s=df[['90% quantile']]**1.5,
                   c=da_granger.sel(Variable=variable).min('lags').values,
               transform=ccrs.PlateCarree(),  alpha=1, vmin=0, vmax=0.2, cmap=cmr.guppy_r)
    for idx2, _ in enumerate(da_granger.location_name.values):
        try:
            ax.text(x=xs.values[idx2], fontdict=font_in, y=ys.values[idx2], transform=ccrs.PlateCarree(),
                    color='black',
                    s=str((da_granger.sel(Variable=variable).isel(location_name=idx2).argmin('lags')+1).values.tolist()))
        except:
            pass
plt.subplots_adjust(hspace=0.2)
cbar = plt.colorbar(p, ax=axs[ :], orientation='vertical', shrink=.6)
cbar.ax.set_ylabel('p-value')
# ax.stock_img()
plt.savefig('figs/map.png', dpi=400,bbox_inches='tight',
            transparent=True,  pad_inches=0)
plt.close()


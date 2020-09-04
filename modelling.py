from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy
import cartopy.feature as cfeature
import cmasher as cmr
import copy
from scipy.stats import shapiro
from xr_tools.tools import get_xr_seq
from scipy import signal

def u_(z, beta, gamma):
    return np.exp(-beta*z/gamma)


def generalize_logistic(t, t0, L, r, xi):
    return L / (1 + xi*np.exp(-r * (t - t0) ))**(1/xi)


def logistic_univariate(t, t0, L, r):

    return L / (1 + np.exp(-r * (t - t0) ))


def logistic_bivariate(inputs, t0, L, r, v):
    t = inputs[0, :]
    pm = inputs[1, :]
    return L / (1 + np.exp(-r * (t - t0) - v*pm))




#  ----- PCA ------ #
to_remove = ['Boise','Boston','Denver','El Paso','Honolulu','Oakland','Omaha','Providence',
             'Sacramento','San Diego','San Fransisco','San Jose','Tucson',
             'Austin','Columbia','Columbus','Dallas','Fort Worth','Houston',
             'Jackson','Little Rock','Manhattan','Memphis','Nashville',
             'Philadelphia','Portland','Richmond','Salem','Salt Lake City',
             'San Antonio','Seattle','Springfield','Tallahassee']
# to_select = ['Baltimore', 'Chicago', 'Indianapolis', 'Hartford', 'Detroit']
filepath = 'data/meteorological_variables_daily_mean.csv'
df = pd.read_csv(filepath)

ds = df.set_index(['time', 'location_name'])
ds = ds.to_xarray()
ds = ds.assign_coords(time=pd.to_datetime(ds.time.values))
to_select = ds.location_name.values.tolist()
to_select = list(set(to_select).difference(to_remove))
# ds = ds.sel(location_name=to_select)
ds_total = ds.copy()
variables = [#'density',
             # 't2m',
             # 'uvb',
             # 'fdir',
             'median_no2',
             'median_pm25', 'median_pm10',
             'covid_cases',
             'covid_cases_first_derivative',
             'covid_cases_second_derivative',
             'covid_cases_first_derivative_smooth',
             'covid_cases_second_derivative_smooth']
predictors = copy.deepcopy(variables)
predictors.remove('covid_cases' )
predictors.remove('covid_cases_second_derivative')
predictors.remove('covid_cases_first_derivative')
predictors.remove('covid_cases_first_derivative_smooth')
predictors.remove('covid_cases_second_derivative_smooth')
ds = ds[variables]
ds = ds.stack({'samples': ['location_name', 'time']})
ds = ds.dropna('samples', how='any')


da_total = ds.to_array()
da_total = da_total.transpose('samples', ...)

ds_pred = ds[predictors]
ds_pred = ds_pred.unstack()
da = ds_pred.to_array()
da['loc_n'] = ('location_name'), np.arange(ds_pred.unstack().location_name.values.shape[0])
da = da.stack({'samples': ['location_name', 'time']})
da = da.transpose('samples', ...)
da = da.dropna('samples', how='any')

from statsmodels.multivariate.pca import PCA

pc = PCA(da.values, standardize=True, normalize=True)
pc.plot_rsquare()
plt.show()


pc.plot_scree()
plt.show()

scores = pc.scores
pc.loadings


# Saving
da_scores = da.copy(data=pc.scores)
da_scores = da_scores.rename(variable='pc')
da_scores = da_scores.assign_coords(pc=np.arange(1,da_scores.pc.shape[0]+1))
da_scores = da_scores.unstack()
da_scores = da_scores.drop('loc_n')

da = da.unstack()
da_scores.name = 'predictors'
da_temp = da_total.unstack().sel(variable='covid_cases')
da_temp1 = da_total.unstack().sel(variable='covid_cases_first_derivative')
da_temp2 = da_total.unstack().sel(variable='covid_cases_second_derivative')
da_temp3 = da_total.unstack().sel(variable='covid_cases_first_derivative_smooth')
da_temp4 = da_total.unstack().sel(variable='covid_cases_second_derivative_smooth')
da_temp.name = 'cases'
da_temp1.name = 'cases_first_derivative'
da_temp2.name = 'cases_second_derivative'
da_temp3.name = 'cases_first_derivative_smooth'
da_temp4.name = 'cases_second_derivative_smooth'
da = xr.merge([da_scores,
                      da_temp.drop('variable'),
                      da_temp1.drop('variable'),
                      da_temp2.drop('variable'),
                      da_temp3.drop('variable'),
                      da_temp4.drop('variable')])

# da_scores.to_dataframe().to_csv('data/pc_scores.csv')
# plt.close()

da = da.where(da.cases > 100)

da = da.unstack()
from xr_tools.tools import common_index
from GrangerCausality.tools.causality import GrangerCausality
location_list = []
da
for location in da.location_name.values:
    da_x = da.sel(location_name=location).predictors.copy()
    da_y = da.sel(location_name=location).cases_first_derivative.copy()
    da_x = da_x.where(da.sel(location_name=location).cases > 100, drop=True)
    da_y = da_y.where(da.sel(location_name=location).cases > 100, drop=True)
    #  preprocess: removing long term trends to make ts stationary
    da_y_original = da_y.copy()

    da_x = da_x / np.abs(da_x.max('time'))
    da_x = da_x - da_x.mean('time')
    da_y_s = da_y.rolling(time=30, center=True).mean()
    da_y = da_y - da_y_s
    da_y = da_y - da_y.mean('time')
    da_y = da_y / np.abs(da_y.max('time'))
    da_x.shift(time=12).plot.line(x='time')
    da_y.plot()
    # plt.legend(['smooth', 'original', 'diff'])
    plt.show()

    gc = GrangerCausality(sampledim='time', featuredim='pc', )
    pvalue_array = gc.run(da_x, da_y, detrend=True, granger_test='ssr_ftest',
                          test_stationarity=False)
    location_list.append(pvalue_array)



da_granger = xr.concat(location_list, dim=da.location_name)
da_granger = da_granger.where(da_granger.min('lags') < 0.1, drop=True)
plt.style.use('grayscale')

plt.figure(figsize=[20, 20])
p = da_granger.plot.line(x='lags', col='pc', add_legend=False)
for idx, ax in enumerate(p.axes.flatten()):
    ax.plot(da_granger.lags, da_granger.mean('location_name').isel(pc=idx),
            lw=8)
    ax.axhline(y=0.1, xmin=0, xmax=30, color='red', linestyle='--')
plt.show()

dc = da.cases_first_derivative
dc = dc / np.abs(dc).max('time')
dc_n = dc.sel(location_name=list(
    set(dc.location_name.values.tolist()).difference(da_granger.location_name.values.tolist()))
)
dc = dc.sel(location_name=da_granger.location_name)
dc.plot.line(x='time')
plt.show()
dc_n.plot.line(x='time')
plt.show()

da_granger.location_name



xs = ds_total.sel(location_name=da_granger.location_name).isel(time=0).longitude
ys = ds_total.sel(location_name=da_granger.location_name).isel(time=0).latitude
avg_pol = ds_total.sel(location_name=da_granger.location_name).median_pm25.max('time')
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Orthographic(-80, 35)},
                       figsize=[7.5, 3])

p = ax.scatter(x=xs, y=ys, s=avg_pol.values ** 3 / 110,
               c=da_granger.sel(pc=2).min('lags').values,
           transform=ccrs.PlateCarree(), cmap='viridis',  alpha=1)
for idx, _ in enumerate(da_granger.location_name.values):
    try:
        ax.text(x=xs.values[idx], y=ys.values[idx], transform=ccrs.PlateCarree(),
                       s=str((da_granger.sel(pc=2).argmin('location_name')+1).values.tolist()[idx]))
    except:
        pass
cbar = plt.colorbar(p)
cbar.ax.set_ylabel('p-value')
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(states_provinces, edgecolor='gray')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.gridlines()
ax.set_title('Minimum p-value among all lags for PC1')
# ax.stock_img()
plt.show()



fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Orthographic(-80, 35)},
                       figsize=[7.5, 3])

p = ax.scatter(x=xs, y=ys,
               c=da_granger.sel(pc=1).min('lags').values,
           transform=ccrs.PlateCarree(),  alpha=1, vmin=0, vmax=.2, cmap='RdBu_r')
for idx, _ in enumerate(da_granger.location_name.values):
    try:
        ax.text(x=xs.values[idx], y=ys.values[idx], transform=ccrs.PlateCarree(),
                       s=str((da_granger.sel(pc=1).argmin('location_name')+1).values.tolist()[idx]))
    except:
        pass
cbar = plt.colorbar(p)
cbar.ax.set_ylabel('p-value')
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(states_provinces, edgecolor='gray')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.gridlines()
ax.set_title('Red: PC1 does NOT Grange cause new covid cases. \n Blue: PC1 Grange causes new covid cases.')
# ax.stock_img()
plt.show()







plt.show()
plt.close()
y = da_a.sel(seq=0).dropna('time')
x = da_b.sel(variable='median_pm25', seq=0).dropna('time')
idxs = common_index(x.time.values, y.time.values)
x = x.sel(time=idxs)
y = y.sel(time=idxs)

X = np.column_stack([y.values, x.values])
rs = grangercausalitytests(X, maxlag=25)
(x*20 - 0.3).shift(time=9).plot()
y.plot()
plt.show()

ols = RollingOLS(da_a.sel(seq=0).values,
                 da_b.sel(seq=-10).transpose('time', ...).values,
                 window=30)

ols_fitted = ols.fit()
plt.style.use('bmh')
coef_r2 = da_b.sel(seq=-1, variable='density').copy(data=ols_fitted.rsquared.T)
coef_llf = da_b.sel(seq=-1, variable='density').copy(data=ols_fitted.llf.T)
coef_p = da_b.sel(seq=-1).copy(data=ols_fitted.pvalues.T)
params = da_b.sel(seq=-1).copy(data=ols_fitted.params.T)
params.name = 'Linear coefficient'
params_ = params.where(coef_p < 0.05)
coef_r2 = coef_r2.unstack()

coef_p.plot.line(col='variable', col_wrap=3)
plt.show()
p = params_.plot.line(col='variable', col_wrap=2)
for idx, ax in enumerate(p.axes.flatten()):
    x = params.time.values
    y = params.isel(variable=idx).where(coef_p.isel(variable=idx) >= 0.05).values
    ax.plot(x, y, linestyle='--', alpha=0.2)
    ax.set_title(params['variable'].values[idx])

plt.show()




plt.savefig('figs/Chicago_rolling_corrs.png')
plt.close()
coef_r2.plot()
plt.show()

coef_llf.plot()
plt.show()

da_a.sel(seq=0).plot()
plt.show()
plt.close()
da.predictors

valid_values = da_a.notnull() & da_b.notnull()
if not valid_values.all():
   da_a = da_a.where(valid_values)
   da_b = da_b.where(valid_values)

valid_count = valid_values.sum('time')

demeaned_da_a = da_a - da_a.mean(dim='time')
demeaned_da_b = da_b - da_b.mean(dim='time')
cov = (demeaned_da_a * demeaned_da_b).sum(dim='time') / (valid_count)

da_a_std = da_a.std(dim='time')
da_b_std = da_b.std(dim='time')
corr = cov / (da_a_std * da_b_std)


corr.plot.line(x='seq', col='location_name', col_wrap=4,
              sharey=True)
plt.show()

fix, axs = plt.subplots(2, 1)
da_scores.cases_first_derivative.sel(location_name='Phoenix', seq=-1).plot(ax=axs[0])
da_scores.pc_scores.sel(location_name='Phoenix', pc=1, seq=-1).plot(ax=axs[1])

plt.show()

for seq in da_scores.seq.values:
    da_scores.sel(pc=2, seq=seq).plot.scatter(x='cases_first_derivative',
                                              y='pc_scores',
                                              col='location_name', col_wrap=4,
                                              sharex=False, sharey=False)
    print('Seq: ' + str(seq))
    plt.show()


da_scores = da_scores
da_scores = da_scores.stack({'sample': ['time', 'location_name']})
x = da_scores.pc_scores.sel(pc=2)
y = da_scores.pc_scores.sel(pc=1)
c = da_scores.cases_first_derivative / \
    np.abs(da_scores.cases_first_derivative).max('sample')
x = x.where(np.abs(c) > .1, drop=True)
y = y.where(np.abs(c) > .1, drop=True)
c = c.where(np.abs(c) > .1, drop=True)


plt.style.use('bmh')
plt.scatter(x.sel(seq=-10), y.sel(seq=-10), c=c.sel(seq=-10), alpha=0.8,
             cmap='RdBu')
plt.colorbar()
plt.show()
plt.close()










da_scores['pc_scores'].sel(pc=1).plot.line(x='time')
plt.show()

df_scores = da_scores['pc_scores'].sel(pc=2).mean('time').to_dataframe()
df_scores_stdev = (da_scores['pc_scores'].var('time')**0.5).sel(pc=2).to_dataframe()
df_scores_stdev = df_scores_stdev.reset_index().pivot(index='location_name', columns='pc')
df_scores.reset_index().pivot(columns='pc', index='location_name').sort_values(by=('pc_scores', 2)).plot.bar(yerr=df_scores_stdev)
plt.show()

da_scores = da_scores.stack({'samples': ['location_name', 'time']})
ds_merged = xr.merge([da_scores, ds])
ds_merged = ds_merged.unstack()
plot_ds = ds_merged.sel(pc=4).sortby((ds_merged.sel(pc=4).pc_scores).mean('time'))
plot_ds.plot.scatter(
    x='time', y='covid_cases_second_derivative',
    col='location_name', col_wrap=4, cmap=cmr.rainforest,
    sharey=False, hue='median_pm25', add_guide=True, vmin=0, vmax=30)
plt.savefig('figs/ts_cases_pm25_second.png')
plt.close()
plt.close()
# Plotting
plt.style.use('bmh')
fig, axs = plt.subplots(2, 3, figsize=[10, 6], gridspec_kw={'wspace':.6, 'hspace':.6})
axs[0, 0].hexbin(scores[:, 0], scores[:, 1], alpha=0.5, cmap=cmr.pride,gridsize=[10, 10],
               C=da_total.sel(variable='covid_cases_second_derivative').values,
                  vmin=-10, vmax=10)
axs[0, 0].set_xlabel('PC1')
axs[0, 0].set_ylabel('PC2')
axs[0, 1].hexbin(scores[:, 0], scores[:, 2], alpha=0.5, cmap=cmr.pride,gridsize=[10,10],
               C=da_total.sel(variable='covid_cases_second_derivative').values,
                  vmin=-10, vmax=10)
axs[0, 1].set_xlabel('PC1')
axs[0, 1].set_ylabel('PC3')
axs[0, 2].hexbin(scores[:, 1], scores[:, 2], alpha=0.5, cmap=cmr.pride,gridsize=[10,10],
               C=da_total.sel(variable='covid_cases_second_derivative').values,
                  vmin=-10, vmax=10)
axs[0, 2].set_xlabel('PC2')
axs[0, 2].set_ylabel('PC3')
axs[1, 0].hexbin(scores[:, 0], scores[:, 3], alpha=0.5, cmap=cmr.pride,gridsize=[10,10],
               C=da_total.sel(variable='covid_cases_second_derivative').values,
                  vmin=-10, vmax=10)
axs[1, 0].set_xlabel('PC1')
axs[1, 0].set_ylabel('PC4')

axs[1, 1].hexbin(scores[:, 1], scores[:, 3], alpha=0.5, cmap=cmr.pride,gridsize=[10,10],
               C=da_total.sel(variable='covid_cases_second_derivative').values,
                  vmin=-10, vmax=10)
axs[1, 1].set_xlabel('PC2')
axs[1, 1].set_ylabel('PC4')

p = axs[1, 2].hexbin(scores[:, 2], scores[:, 3], alpha=0.5, cmap=cmr.pride, gridsize=[10,10],
               C=da_total.sel(variable='covid_cases_second_derivative').values,
                      vmin=-10, vmax=10)
axs[1, 2].set_xlabel('PC3')
axs[1, 2].set_ylabel('PC4')
fig.colorbar(p, ax=axs, orientation='vertical')

plt.savefig('figs/pcs.pdf')
plt.close()

pc.plot_rsquare()
plt.show()
da_loadings = xr.DataArray(pc.loadings, dims=['PCs', 'Variable'],
                           coords={'PCs': np.arange(1, 5), 'Variable':da['variable'].values.tolist() })

df_loadings = da_loadings.to_dataframe(name='loadings')
df_loadings = df_loadings.reset_index().set_index('Variable').pivot(columns='PCs')
df_loadings.plot.bar(subplots=True, layout=[4, 1], figsize=[8,10], sharey=True)
plt.show()


import seaborn as sn
res = sn.heatmap(df_loadings, annot=True,  fmt='.2f', cmap=cmr.seasons, #
                 cbar_kws={'label':'PC Loading'})

res.set_xlabel('PCs')

plt.show()

fig, ax = plt.subplots(1)
p = ax.imshow(np.abs(pc.loadings), cmap=cmr.freeze)
ax.text()
ax.set_xticklabels( da['variable'].values.tolist())
ax.set_yticks([0,1,2,3])
ax.set_xticks([0,1,2,3])
ax.set_yticklabels(np.arange(1, 5))
ax.set_ylabel('PC')
fig.colorbar(p)
plt.show()
proj = pc.project(ncomp=2)
proj.shape
plt.show()


# -----  old ------ #
df = pd.read_csv(filepath)
ds = df.set_index(['time', 'location_name']).to_xarray()
ds = ds.assign_coords(time=pd.to_datetime(ds.time.values))
da = ds['median_pm25']
da = da.assign_coords(latitude=('location_name', ds.isel(time=0)['latitude'].values))
da = da.assign_coords(longitude=('location_name', ds.isel(time=0)['longitude'].values))

da_avg = da.mean('time')
pop = ds.population.isel(time=0).values
pop = pop/np.max(pop)


# --- Plotting

states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Orthographic(-80, 35)},
                       figsize=[7.5, 3])

p = ax.scatter(x=da.longitude.values, y=da.latitude.values, c=da_avg,
           transform=ccrs.PlateCarree(), cmap=cmr.gem_r, s=pop*300, alpha=0.65)
cbar = plt.colorbar(p)
cbar.ax.set_ylabel('pm2.5 concentration')
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(states_provinces, edgecolor='gray')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.gridlines()
ax.set_title('Locations colored by PM2.5 and scaled by population size')
# ax.stock_img()
plt.show()

# --- Fitting models
outbreak_phase = 0

ds_fit = ds.covid_cases.where(ds.outbreak_phase == outbreak_phase, drop=True)
ds_fit.stack({'samples': ['location_name', 'time']})
plt.style.use('bmh')
pm = ds['median_pm25'].shift(time=0).where(ds.outbreak_phase == outbreak_phase, drop=True)
media = pm.mean('time')
pm = pm.assign_coords(media=('location_name', media))
pm = pm.sortby(media)
ds_fit = ds_fit.sortby(media)


fig, axs = plt.subplots(2, 1)
p = axs[1].pcolormesh(pm.values, cmap=cmr.rainforest, vmax=80, vmin=10)
axs[0].plot(ds_fit.mean('time').values)
axs[0].semilogy(True)
axs[0].set_ylabel('Pop')
axs[1].set_xlabel('Cidade')
axs[1].set_ylabel('Dia na fase 0')
cbar = plt.colorbar(p, orientation='horizontal', shrink=0.5)
cbar.ax.set_xlabel('PM25')
plt.show()

plt.show()
ds_fit.plot(x='time',col='location_name', col_wrap=5, sharey=False)
plt.savefig(f'figs/panel_phase{outbreak_phase}.pdf')
plt.close()

initial_cases = 5
locations = ds_fit.location_name.values


def run_model(ds_fit, pm, locations, model_type='univariate'):



    params_list = []
    locations_fitted = []
    lats = []
    lons = []
    pms = []
    shapiro_ps = []
    pops = []
    for location in locations:
        y_var = ds_fit.sel(location_name=location)
        y_var = y_var.where(y_var > 5, drop=True)

        time_idx = [kk for kk, _ in enumerate(y_var.time.values)]
        if model_type == 'univariate':
            params_names = ['t0', 'L', 'r']
            model = logistic_univariate
            bounds = ([0, 0, 0], [len(time_idx), 1e7, 1])
        elif model_type == 'bivariate':
            params_names = ['t0', 'L', 'r', 'v']
            model = logistic_bivariate
            bounds = ([0, 0, -5, -1], [len(time_idx), 1e7, 5, 1])
        elif model_type == 'generalized':
            params_names = ['t0', 'L', 'r', 'xi']
            model = generalize_logistic
            bounds = ([0, 0, -2, 1e-6], [len(time_idx), 1e7, 2, 2])
        try:
            if model_type == 'bivariate':
                input = np.stack([np.array(time_idx), pm.sel(location_name=location, time=y_var.time).dropna('time').values])
            else:
                input = np.array(time_idx)
            params, _ = curve_fit(model, input, y_var, bounds=bounds)
            y_pred = y_var.copy(data=model(input, *params))
            # y_pred.plot()
            # y_var.plot()
            # plt.legend(['model', 'real'])
            # plt.show()
            err = (y_pred - y_var).values
            shap = shapiro(err)[1]

        except:
            print(f'{location} fitting failed')
        else:


            shapiro_ps.append(shap)

            params_list.append(params)
            locations_fitted.append(location)
            lats.append(float(ds.latitude.isel(time=0).sel(location_name=location).values))
            lons.append(float(ds.longitude.isel(time=0).sel(location_name=location).values))
            pms.append(pm.sel(location_name=location).mean().values)
            pops.append(ds.population.isel(time=0).sel(location_name=location).values)



    da_fitted = xr.DataArray(params_list, dims=['location_name', 'params'],
                             coords={'location_name': locations_fitted, 'params': params_names})
    da_fitted = da_fitted.assign_coords(latitude=('location_name', lats))
    da_fitted = da_fitted.assign_coords(longitude=('location_name', lons))
    da_fitted = da_fitted.assign_coords(pm=('location_name', pms))
    da_fitted = da_fitted.assign_coords(pop=('location_name', pops))
    da_fitted = da_fitted.assign_coords(p=('location_name', shapiro_ps))
    return da_fitted

da_fitted = run_model(ds_fit, pm, locations, model_type='generalized')
# --- Plotting
param = 'xi'
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Orthographic(-80, 35)},
                       figsize=[7.5, 3])

p = ax.scatter(x=da_fitted.longitude.values, vmax=1e-4,y=da_fitted.latitude.values, c=da_fitted.sel(params=param),
           transform=ccrs.PlateCarree(), cmap=cmr.rainforest, alpha=1, s=0.00005*da_fitted.pm.values**4)
cbar = plt.colorbar(p)
cbar.ax.set_ylabel(param)
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(states_provinces, edgecolor='gray')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.gridlines()
plt.show()
da_fitted.pm

# ax.set_title('Locations colored by PM2.5 and scaled by population size')
da_fitted.shift()

plt.scatter(da_fitted.pm.values, da_fitted.sel(params='xi').values)
plt.semilogx(True)
plt.style.use('bmh')
for loc in da_fitted.location_name.values:
    plt.text(x=da_fitted.pm.sel(location_name=loc).values,
             y=da_fitted.sel(params='xi', location_name=loc).values, s=loc)
plt.show()
infected_fraction=100*da_fitted.sel(params='L').values/da_fitted.pop.values
p = plt.scatter(da_fitted.pm.values, da_fitted.sel(params='t0').values,
            c=da_fitted.p.values, alpha=0.8, cmap=cmr.ember, vmax=0.01)
cbar = plt.colorbar(p)
cbar.ax.set_ylabel('shapiro')
plt.xlabel('pm')
plt.ylabel('Inflex')
plt.show()

threshold = 0.01
to_plot = da_fitted.where(da_fitted.p < threshold, drop=True)
media = media.where(da_fitted.p < threshold, drop=True)
to_plot = to_plot.sortby(media)
pm = pm.sortby(media)

param1='r'
param2='t0'
param3 = 'xi'
fig, axs = plt.subplots(5, 1, figsize=[8, 8],
                        gridspec_kw={'height_ratios':
                                         [0.24,0.24, 0.24, 0.24,.03]})

axs[0].bar(x=np.arange(to_plot.location_name.shape[0]),
           height=to_plot.sel(params=param1).values,
           align='edge', width=1)
axs[0].set_ylim([0, 0.22])
axs[0].set_xlim([0, 17])
# axs[0].hlines(np.mean(to_plot.sel(params=param1).values[0:14]),
#               xmin=0, xmax=13)
# axs[0].hlines(np.mean(to_plot.sel(params=param1).values[14:-1]),
#               xmin=14, xmax=30)
axs[0].set_ylabel(param1)
axs[1].bar(x=np.arange(to_plot.location_name.shape[0]),
           height=to_plot.sel(params=param2).values,
           align='edge', width=1)
axs[1].set_ylabel(param2)
axs[1].set_xlim([0, 17])

p = axs[3].pcolormesh(pm.sel(location_name=to_plot.location_name).values, cmap=cmr.gem, vmax=60, vmin=10)
axs[3].set_xlabel('Cidade')
axs[3].set_ylabel('Dia na fase 0')
axs[3].plot(media.sortby(media), color='red')

axs[2].bar(x=np.arange(to_plot.location_name.shape[0]),
           height=to_plot.sel(params=param3).values,
           align='edge', width=1)
axs[2].set_ylabel(param3)
axs[2].set_xlim([0, 17])
axs[2].set_yscale('log')
cbar = fig.colorbar(p, orientation='horizontal', shrink=0.5,
                    cax=axs[4])
cbar.ax.set_xlabel('PM25')
plt.show()


plt.scatter(da_fitted.sel(params='r').values, da_fitted.sel(params='t0').values,
            c=da_fitted.pm.values, cmap=cmr.rainforest)
plt.show()

plt.scatter(da_fitted.pm.values, (da_fitted.sel(params='r')).values)
plt.show()

infected_fraction=100*to_plot.sel(params='L').values/to_plot.pop.values

p = plt.scatter(to_plot.pm.values, to_plot.sel(params='t0').values,
            c=infected_fraction, alpha=0.8, cmap=cmr.ember, vmax=1)
cbar = plt.colorbar(p)
cbar.ax.set_ylabel('infected_fraction')
plt.xlabel('pm')
plt.ylabel('Inflex')
plt.show()

p = plt.scatter(to_plot.sel(params='t0').values, to_plot.sel(params='r').values,
            c=infected_fraction, alpha=0.8, cmap=cmr.ember, vmax=1)
cbar = plt.colorbar(p)
cbar.ax.set_ylabel('Fraction infected in first stage (%)')
plt.xlabel('Inflex')
plt.ylabel('Growth')
plt.show()

p = plt.scatter(to_plot.sel(params='t0').values, to_plot.sel(params='r').values,
            c=to_plot.pm.values, alpha=0.8, cmap=cmr.ember, vmax=40)
cbar = plt.colorbar(p)
cbar.ax.set_ylabel('PM25')
plt.xlabel('Inflex')
plt.ylabel('Growth')
plt.show()

plt.plot([u_(z, 0.1, 0.1) for z in np.arange(0, 100)])
plt.show()

# --- Stacked
from sklearn.preprocessing import MinMaxScaler
ds_fit = ds.covid_cases.where(ds.outbreak_phase == outbreak_phase, drop=True)
pop = ds.population.where(ds.outbreak_phase == outbreak_phase, drop=True)
#
# scaler = MinMaxScaler()
# scaler = scaler.fit(X=ds_fit.values)
ds_scaled = ds_fit.copy(data=ds_fit/pop)
ds_fit_stacked = ds_scaled.stack({'samples': ['location_name', 'time']})
time_samples = [x[1] for x in ds_fit_stacked.samples.values]
city = ds_fit_stacked.unstack().sel(location_name='Tallahassee').dropna('time')
plt.scatter(y=ds_fit_stacked.values, x=time_samples, alpha=0.2)
plt.scatter(y=city,
            x=city.time.values, color='red')


ds_fit_stacked.expand_dims

plt.show()
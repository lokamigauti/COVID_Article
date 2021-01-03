import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np


def logistic(t, t0, L, r):
    return L / (1 + np.exp(-r * (t - t0)))

scenarios = {'PM25 Lag 14': dict(
slope_inflex = -0.3218,
std_inflex = 0.1356,
slope_growth = 0.0017211,
std_growth = 0.0007892,
),
'PM25 Lag 5': dict(
slope_inflex = -0.3726,
std_inflex = 0.1413,
slope_growth = 0.0019073,
std_growth = 0.0008379,
),

'PM10 Lag 14': dict(
slope_inflex = -0.7737,
std_inflex = 0.6820,
slope_growth = 0.003349,
std_growth = 0.002605,
),

'PM10 Lag 5': dict(
slope_inflex = -0.6569,
std_inflex = 0.6487,
slope_growth = 0.003191,
std_growth = 0.002428,
),
'Boston1': dict(
slope_inflex = 0,
std_inflex = 0,
slope_growth = 0.096,
std_growth = 0,
),
'Boston2': dict(
slope_inflex = 0,
std_inflex = 0,
slope_growth = 0.108,
std_growth = 0,
)
}


def plot_scenario(scenario_name):
    scenario = scenarios[scenario_name]
    dpm = 20
    std_inflex = scenario['std_inflex']
    slope_inflex = scenario['slope_inflex']
    slope_growth = scenario['slope_growth']
    std_growth = scenario['std_growth']
    # base_t0 = 30
    base_t0 = 46.89
    base_r = 0.096
    # base_r = 0.16735
    new_t0 = base_t0 + slope_inflex * dpm
    new_t0 = base_t0
    new_t0_std1 = base_t0 + slope_inflex * dpm + std_inflex * dpm
    new_t0_std2 = base_t0 + slope_inflex * dpm - std_inflex * dpm
    new_t0_std1 = 0
    new_t0_std2 = 0
    new_r = base_r + slope_growth * dpm
    new_r_std1 = base_r + slope_growth * dpm + std_growth * dpm
    new_r_std2 = base_r + slope_growth * dpm - std_growth * dpm
    new_r = slope_growth
    ts = np.arange(0, 100, 1)

    y1 = logistic(ts, base_t0, 100, base_r)

    y2 = logistic(ts, new_t0, 100, new_r)
    y2_std1 = logistic(ts, new_t0_std1, 100, new_r_std2)

    y2_std2 = logistic(ts, new_t0_std2, 100, new_r_std1)


    plt.style.use('seaborn-white')
    plt.plot(ts[1:], np.diff(y1))
    plt.plot(ts[1:], np.diff(y2))
    # plt.plot(ts[1:], np.diff(y2_std1), color='k', linestyle='--', alpha=0.5)
    # plt.plot(ts[1:], np.diff(y2_std2), color='k', linestyle='--',  alpha=0.5)
    plt.legend(['Scenario 1', 'Scenario 2', 'Scenario 2 + std. deviation', 'Scenario 2 - std. deviation'])
    plt.ylabel('Daily new cases (% of total population)')
    plt.xlabel('Time since first case (days)')
    plt.savefig(f'figs/new_cases_{scenario_name}.pdf', transparent=True, pad_inches=.2,)
    plt.close()
    plt.style.use('seaborn-white')
    plt.plot(ts, (y1))
    plt.plot(ts, (y2))
    # plt.plot(ts, (y2_std1), color='k', linestyle='--', alpha=0.5)
    # plt.plot(ts, (y2_std2), color='k', linestyle='--',  alpha=0.5)
    plt.legend(['Scenario 1', 'Scenario 2',]) #'Scenario 2 + std. deviation', 'Scenario 2 - std. deviation'])
    plt.ylabel('Total cases (% of total population)')
    plt.xlabel('Time since first case (days)')
    plt.savefig(f'figs/total_cases_{scenario_name}.pdf', transparent=True, pad_inches=.2,)
    plt.close()

plt.style.use('ggplot')
for scenario_name in ['Boston2']:
    plot_scenario(scenario_name)





ds = pd.read_csv('data/corr_df.csv')
ds = ds.set_index(['mavgs', 'lags'])


ds = ds.to_xarray()
ds = ds.rename({'meancorr': 'Mean correlation'})
ds = ds.rename({'mediancorr': 'Median correlation'})
fig, ax = plt.subplots()

ratio = ds['Median correlation']/ ds['sdcorr']

ratio.plot( edgecolor='k',vmin=0,ax=ax, cbar_kwargs=dict(orientation='horizontal'), cmap='YlGnBu')
# ds['sdcorr'].plot( ax=ax,edgecolor='k', add_colorbar=False)
ax.set_aspect('equal')
ds.sel(lags=10, mavgs=2)
# p = ds['sdcorr'].plot.contour(levels=8,hatch=['+','/'],ax=ax,colors='k', inline=1, linetype='--',alpha=0.7,
#                               )
# ax.clabel(p, inline=1)
ax.set_xticks(ds.lags.values)
ax.set_yticks(ds.mavgs.values)

ax.set_xticklabels(ds.lags.values)
ax.set_xlabel('Lags (days)')
ax.set_ylabel('Backward moving windows (days)')
plt.title('Positive only correlation scaled by standard deviation')
plt.tight_layout()
plt.show()
# ds = ds.set_index(['location_name', 'time'])
# ds = ds.drop(['latitude', 'longitude'], axis=1)
# ds = ds.to_xarray()
# ds = ds.assign_coords(time=[pd.Timestamp(x) for x in ds.time.values])
# lr = LogisticRegression()
# day_idx = np.arange(ds['covid_cases'].sel(location_name='Los Angeles').dropna('time').time.shape[0])
# lr.fit(day_idx.reshape(-1,1), ds['covid_cases'].sel(location_name='Los Angeles').dropna('time').values.reshape(-1, 1))
# plt.scatter(day_idx, lr.predict(day_idx.reshape(-1,1)))
# plt.show()
# # ds = ds.resample(time='3D').mean()
# ds['covid_trend'] = ds['covid_cases'].diff('time', n=1)
# p = ds.plot.scatter(x='time',y='covid_trend', col='location_name', col_wrap=2, aspect=4)
# for i, ax in enumerate(p.axes.flat):
#     ax2 = ax.twinx()
#     ds['median_mp25'].isel(location_name=i).plot(ax=ax2, color='red')
#     ax2.set_title(None)
#
# plt.legend(['Covid cases', 'MP2.5 concentration'])
# p.fig.tight_layout()
# plt.show()
#
# from sklearn.linear_model import LinearRegression
# x=ds.where(ds.is_quarentined == 1, drop=True).mean('time').dropna(dim='location_name', how='any')['median_mp25']
# y=ds.where(ds.is_quarentined == 1, drop=True).mean('time').dropna(dim='location_name', how='any')['covid_trend']
# lm = LinearRegression()
# lm = LinearRegression()
# lm = lm.fit(x.values.reshape([-1,1]), y.values.reshape([-1,1]))
# predicted = lm.predict(x.values.reshape([-1,1]))
# score = lm.score(x.values.reshape([-1,1]), y.values.reshape([-1,1]))
# plt.style.use('seaborn')
# ds.where(ds.is_quarentined == 1, drop=True).mean('time').plot.scatter(x='median_mp25', y='covid_trend', color='black')
# plt.plot(x,predicted, color='black', linestyle='-')
# plt.text(27.5, 20, 'R² = ' + str(round(score,2)))
# plt.xlabel('Median of PM 2.5 averaged during the quarentine period')
# plt.ylabel('New covid cases per day')
# plt.show()
#
# plt.boxplot([y.where(x<20).dropna('location_name'), y.where(x>20).dropna('location_name')])
# plt.xticks(ticks=[1,2],labels=['PM < 20', 'PM >20'])
# plt.ylabel('Daily increase rate of Covid cases')
# plt.xlabel('PM median during quarentine period')
# plt.show()
#
# ds['uvb_scaled'] = ds['uvb'] / ds['fdir']
# ds['uvb_scaled'] = ds['uvb_scaled'].where(ds['uvb'] < ds['fdir'], 0)
#
# plt.show()
#
# def make_patch_spines_invisible(ax):
#     ax.set_frame_on(True)
#     ax.patch.set_visible(False)
#     for sp in ax.spines.values():
#         sp.set_visible(False)
#
#
# vars = ['fdir', 'uvb', 't2m']
# for var in vars:
#     for loc in ds.location_name.values:
#         plt.style.use('seaborn')
#
#         fig, ax = plt.subplots(1,1, figsize=[12,6])
#         fig.subplots_adjust(right=0.75)
#         plt.style.use('default')
#         ax2 = ax.twinx()
#         ax3 = ax.twinx()
#         ax3.spines["right"].set_position(("axes", 1.2))
#         make_patch_spines_invisible(ax3)
#         ax3.spines["right"].set_visible(True)
#
#         ds['covid_trend'].sel(location_name=loc).plot(ax=ax, color='blue', )
#         ds[var].sel(location_name=loc).plot(ax=ax2, color='red', )
#         ds['median_mp25'].sel(location_name=loc).plot(ax=ax3, color='black', )
#
#         ax.set_ylabel('Trend of the number of cases', color='blue')
#         ax2.set_ylabel(f'{var}', color='red')
#         ax3.set_ylabel('Median PM2.5 ', color='black')
#
#         ax2.grid(False)
#         ax3.grid(False)
#         # fig.legend(['covid_trend', 'median_mp25'])
#         plt.savefig(f'figs/{loc}_{var}.pdf')
#         plt.close()
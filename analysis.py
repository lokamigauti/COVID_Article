import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
ds = pd.read_csv('data/corr_df.csv')
ds = ds.set_index(['mavgs', 'lags'])


ds = ds.to_xarray()
ds = ds.rename({'meancorr': 'Mean correlation'})
ds = ds.rename({'mediancorr': 'Median correlation'})
fig, ax = plt.subplots()

ds['Mean correlation'].plot( edgecolor='k',ax=ax, cbar_kwargs=dict(orientation='horizontal'), cmap='seismic')
# ds['sdcorr'].plot( ax=ax,edgecolor='k', add_colorbar=False)
ax.set_aspect('equal')
ds.sel(lags=10, mavgs=2)
p = ds['sdcorr'].plot.contour(levels=4,hatch=['+','/'],ax=ax,colors='k', inline=1, linetype='--',alpha=0.5,
                              )
ax.clabel(p, inline=1)
ax.set_xticks(ds.lags.values)
ax.set_yticks(ds.mavgs.values)

ax.set_xticklabels(ds.lags.values)
ax.set_xlabel('Lags (days)')
ax.set_ylabel('Backward moving windows (days)')
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
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
ds = pd.read_csv('data/meteorological_variables_daily_mean.csv')
ds = ds.set_index(['location_name', 'time'])
ds = ds.drop(['latitude', 'longitude'], axis=1)
ds = ds.to_xarray()
ds = ds.assign_coords(time=[pd.Timestamp(x) for x in ds.time.values])
# ds = ds.resample(time='3D').mean()
ds['covid_trend'] = ds['covid_cases'].diff('time', n=1)
p = ds.plot.scatter(x='time',y='covid_trend', col='location_name', col_wrap=2, aspect=4)
for i, ax in enumerate(p.axes.flat):
    ax2 = ax.twinx()
    ds['median_mp25'].isel(location_name=i).plot(ax=ax2, color='red')
    ax2.set_title(None)

plt.legend(['Covid cases', 'MP2.5 concentration'])
p.fig.tight_layout()
plt.show()

ds.mean('time').plot.scatter(x='uvb', y='covid_cases')


ds['uvb_scaled'] = ds['uvb'] / ds['fdir']
ds['uvb_scaled'] = ds['uvb_scaled'].where(ds['uvb'] < ds['fdir'], 0)

plt.show()

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


var = 'fdir'
for loc in ds.location_name.values:
    plt.style.use('seaborn')

    fig, ax = plt.subplots(1,1, figsize=[12,6])
    fig.subplots_adjust(right=0.75)
    plt.style.use('default')
    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.2))
    make_patch_spines_invisible(ax3)
    ax3.spines["right"].set_visible(True)

    ds['covid_trend'].sel(location_name=loc).plot(ax=ax, color='blue', )
    ds[var].sel(location_name=loc).plot(ax=ax2, color='red', )
    ds['median_mp25'].sel(location_name=loc).plot(ax=ax3, color='black', )

    ax.set_ylabel('Trend of the number of cases', color='blue')
    ax2.set_ylabel(f'{var}', color='red')
    ax3.set_ylabel('Median PM2.5 ', color='black')

    ax2.grid(False)
    ax3.grid(False)
    # fig.legend(['covid_trend', 'median_mp25'])
    plt.savefig(f'figs/{loc}_{var}.pdf')
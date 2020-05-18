import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
locations_x, locations_y = np.meshgrid(np.arange(0,10,1), np.arange(0, 10, 1))
from scipy.signal import gaussian
plt.style.use('seaborn')



tempo_onde_uma_pessoa_pode_transmitir = 15
taxa_de_transmissao_por_pessoa = 2.6 / 8.6e6
infection_rate = taxa_de_transmissao_por_pessoa / tempo_onde_uma_pessoa_pode_transmitir
# infection_rate = 0.5e-6


def run_simulation(particulate, dt=1/24, ndays=365, popsize=8.6e6, removal_rate=1e-3, infection_rate=2e-2):
    nt = int(ndays / dt)
    I = np.zeros(nt)
    R = np.zeros(nt)
    S = np.zeros(nt)

    I[0] = 11  # first individual
    R[0] = 0  # removed
    S[0] = popsize
    alpha = removal_rate

    beta = infection_rate + infection_rate * particulate

    for t in range(nt - 1):
        R[t + 1] = R[t] + dt * alpha*I[t]
        I[t + 1] = I[t] + dt * (
                beta[t]*S[t]*I[t] -alpha*I[t])
        S[t + 1] = S[t] - dt * beta[t]*S[t]*I[t]
    return R, I, S, beta


ndays = 60
dt = 1/24
nt = int(ndays / dt)
times = np.arange(0, ndays, dt)  # time in days

particulate = np.zeros(nt)
particulate[10:200] = 1
# particulate = 10 * gaussian(nt, std = 100)
particulate_control = np.zeros(nt)
plt.plot(times, particulate)
plt.ylabel('Concentration of particulate')
plt.xlabel('Time since outbreak')

plt.show()

control_R, control_I, control_S, beta_control = run_simulation(particulate_control, infection_rate=infection_rate,
                                                 dt=dt, ndays=ndays,
                                                 removal_rate=2e-1)

R, I, S, beta = run_simulation(particulate, dt=dt, infection_rate=infection_rate,
                               ndays=ndays, removal_rate=2e-1  )

I = xr.DataArray(I, dims=['time'], coords=dict(time=times))
R = xr.DataArray(R, dims=['time'], coords=dict(time=times))
S = xr.DataArray(S, dims=['time'], coords=dict(time=times))
control_I = xr.DataArray(control_I, dims=['time'], coords=dict(time=times))
control_R = xr.DataArray(control_R, dims=['time'], coords=dict(time=times))
control_S = xr.DataArray(control_S, dims=['time'], coords=dict(time=times))

plt.style.use('seaborn')
plt.figure(figsize=[20, 20])
da_control = xr.concat([ control_I, control_R, control_S],
               pd.Index([ 'control_I', 'control_R', 'control_S'], name='Population type'))
da_experiment = xr.concat([ I, R, S],
               pd.Index([ 'I', 'R', 'S'], name='Population type'))
da = xr.concat([da_experiment, da_control], pd.Index(['Experiment', 'Control'], name='Experiment type'))
da.plot.line(x='time', row='Experiment type')
plt.ylabel('Population #')
plt.xlabel('Time since outbreak started (days)')
plt.show()

(da_experiment - da_control.values).plot.line(x='time')

I.differentiate('time').plot()
control_I.differentiate('time').plot()

plt.legend(['Experiment', 'Control'])
plt.show()
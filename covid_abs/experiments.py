"""
Common code for simulation experiments in batch
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from covid_abs.common import *
from covid_abs.agents import *
from covid_abs.abs import *
from covid_abs.graphics import color1, color3, legend_ecom


def plot_mean_std(ax, mean, std, legend, color=None):
    l = len(mean)
    lb = [mean[k] - std[k] for k in range(l)]
    ub = [mean[k] + std[k] for k in range(l)]

    ax.fill_between(range(l), ub, lb,
                     color=color, alpha=.5)
    # plot the mean on top
    ax.plot(mean, color, label=legend)


def plot_batch_results(df):
    """
    Plot the results of a batch executions contained in the given DataFrame
    :param df: Pandas DataFrame returned by batch_experiment method
    """
    metrics = df['Metric'].unique()

    health_metrics = [k for k in metrics if "Q" not in k and "Asym" not in k]
    ecom_metrics = [k for k in metrics if "Q" in k]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[20, 5])

    ax[0].set_title('Average Contagion Evolution')
    ax[0].set_xlabel("Nº of Days")
    ax[0].set_ylabel("% of Population")

    for col in health_metrics:
      means = df[(df["Metric"] == col)]['Avg'].values
      std = df[(df["Metric"] == col)]['Std'].values
      plot_mean_std(ax[0], means, std, legend=col, color=color1(col))

    handles, labels = ax[0].get_legend_handles_labels()
    lgd = ax[0].legend(handles, labels, loc='top right')

    mmax = 0.0
    mmin = np.inf
    smax = 0
    smin = np.inf

    for col in ecom_metrics:
      val = df[(df["Metric"] == col)]['Avg'].values
      tmp = int(np.max(val))
      mmax = np.max([mmax, tmp])
      tmp = np.min(val)
      mmin = np.min([mmin, tmp])
      val = df[(df["Metric"] == col)]['Std'].values
      tmp = np.max(val)
      smax = np.max([smax, tmp])
      tmp = np.min(val)
      smin = np.min([smin, tmp])

    ax[1].set_title('Average Economical Impact')
    ax[1].set_xlabel("Nº of Days")
    ax[1].set_ylabel("Wealth")

    for col in ecom_metrics:
      means = df[(df["Metric"] == col)]['Avg'].values
      n_mean = np.interp(means, (mmin, mmax), (0, 1))
      std = df[(df["Metric"] == col)]['Std'].values
      n_std = np.interp(std, (smin, smax), (0, 1))
      ax[1].plot(n_mean, label=legend_ecom[col])
      #std = np.log10(df[(df["Metric"] == col)]['Std'].values)
      #plot_mean_std(ax[1], n_mean, n_std, color=color3(col))

    handles, labels = ax[1].get_legend_handles_labels()
    lgd = ax[1].legend(handles, labels, loc='top left')


def batch_experiment(experiments, iterations, file, simulation_type=Simulation, **kwargs):
    """
    Execute several simulations with the same parameters and store the average statistics by iteration

    :param experiments: number of simulations to be performed
    :param iterations: number of iterations on each simulation
    :param file: filename to store the consolidated statistics
    :param simulation_type: Simulation or MultiPopulationSimulation
    :param kwargs: the parameters of the simulation
    :return: a Pandas Dataframe with the consolidated statistics by iteration
    """
    verbose = kwargs.get('verbose', None)
    rows = []
    columns = None
    for experiment in range(experiments):
        if verbose == 'experiments':
            print('Experiment {}'.format(experiment))
        sim = simulation_type(**kwargs)
        sim.initialize()
        if columns is None:
            statistics = sim.get_statistics(kind='all')
            columns = [k for k in statistics.keys()]
        for it in range(iterations):
            if verbose == 'iterations':
                print('Experiment {}\tIteration {}'.format(experiment, it))
            sim.execute()
            statistics = sim.get_statistics(kind='all')
            statistics['iteration'] = it
            rows.append(statistics)

    df = pd.DataFrame(rows, columns=[k for k in statistics.keys()])

    rows2 = []
    for it in range(iterations):
        df2 = df[(df['iteration'] == it)]
        for col in columns:
            row = [it, col, df2[col].values.min(), df2[col].values.mean(),df2[col].values.std(), df2[col].values.max()]
            rows2.append(row)

    print(rows2)

    df2 = pd.DataFrame(rows2, columns=['Iteration', 'Metric', 'Min', 'Avg', 'Std', 'Max'])

    df2.to_csv(file, index=False)

    return df2

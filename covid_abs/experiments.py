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


def plot_batch_results(df, health_metrics=('Susceptible', 'Infected', 'Hospitalization', 'Severe', 'Recovered_Immune', 'Death'),
                       ecom_metrics=('Q1', 'Q2', 'Q3', 'Q4', 'Q5')):
    """
    Plot the results of a batch executions contained in the given DataFrame
    :param ecom_metrics:
    :param health_metrics:
    :param df: Pandas DataFrame returned by batch_experiment method
    """

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
        # std = np.log10(df[(df["Metric"] == col)]['Std'].values)
        # plot_mean_std(ax[1], n_mean, n_std, color=color3(col))

    handles, labels = ax[1].get_legend_handles_labels()
    lgd = ax[1].legend(handles, labels, loc='top left')


def plot_graph_batch_results(df, **kwargs):
    """
    Plot the results of a batch executions contained in the given DataFrame
    :param ecom_metrics:
    :param health_metrics:
    :param df: Pandas DataFrame returned by batch_experiment method
    """
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                   AutoMinorLocator)
    A1 = ['Q1','Q2','Q3','Q4','Q5']

    health_metrics=('Susceptible', 'Infected', 'Hospitalization', 'Severe', 'Recovered_Immune', 'Death')
    ecom_metrics=('Q1', 'Q2', 'Q3', 'Q4', 'Q5','Business','Government')

    health_legend={'Susceptible': '$S_t$ - Susceptible', 'Infected':'$I_t$ - Infected', 
                   'Hospitalization':'$S^H_t$ - Hospitalization', 'Severe':'$I^S_t$ - Severe', 
                   'Recovered_Immune':'$R_t$ - Recovered', 'Death':'$D_t$ - Death'}

    ecom_legend = {'A1': '$W^{A1}_t$ - People',
              'Business': '$W^{A3}_t$ - Business',
              'Government': '$W^{A4}_t$ - Government',}

    colors = {'A1': 'purple',
              'Business': 'red',
              'Government': 'brown',}
    epidem = kwargs.get('epidem', True)

    iterations = max(df['Iteration'].values) + 1

    tick_unit = kwargs.get('tick_unit', 72)

    tickslabels = [str(i // 24) for i in range(0, iterations, tick_unit)]

    fig, ax = plt.subplots(nrows=1, ncols=1 if not epidem else 2, figsize=[15, 4])

    if epidem:
      ep_ax = ax[0]
      ep_ax.set_title('Average Epidemiological Evolution')
      ep_ax.set_xlabel("Nº of Days")
      ep_ax.set_ylabel("% of Population")
      ep_ax.set_xlim((0, iterations))
      ep_ax.xaxis.set_major_locator(MultipleLocator(tick_unit))
      ep_ax.set_xticklabels(tickslabels)

      for col in health_metrics:
          means = df[(df["Metric"] == col)]['Avg'].values
          std = df[(df["Metric"] == col)]['Std'].values
          plot_mean_std(ep_ax, means, std, legend=health_legend[col], color=color1(col))

      handles, labels = ep_ax.get_legend_handles_labels()
      lgd = ep_ax.legend(handles, labels, loc=2, bbox_to_anchor=(1, 1))

    ec_ax = ax[1] if epidem else ax

    ec_ax.set_title('Average Economical Impact')
    ec_ax.set_xlabel("Nº of Days")
    ec_ax.set_ylabel("% of GDP")
    ec_ax.set_xlim((0, iterations))
    ec_ax.xaxis.set_major_locator(MultipleLocator(tick_unit))
    ec_ax.set_xticklabels(tickslabels)

    for col in ecom_legend.keys():
      if col == 'A1':
        means = np.zeros(iterations)
        std = np.zeros(iterations)
        for m2 in A1: 
          means += df[(df["Metric"] == m2)]['Avg'].values  
          std += df[(df["Metric"] == m2)]['Std'].values
      else:  
        means = df[(df["Metric"] == col)]['Avg'].values
        std = df[(df["Metric"] == col)]['Std'].values
      l = len(means)
      lb = [means[k] - std[k] for k in range(l)]
      ub = [means[k] + std[k] for k in range(l)]

      ec_ax.fill_between(range(l), ub, lb, color=colors[col], alpha=.4)
      ec_ax.plot(range(l), means, colors[col], label=ecom_legend[col])

    handles, labels = ec_ax.get_legend_handles_labels()
    lgd = ec_ax.legend(handles, labels, loc=2, bbox_to_anchor=(1, 1))

    plt.tight_layout()


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
        try:
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
        except Exception as ex:
            print("Exception occurred in experiment {}: {}".format(experiment, ex))

    df = pd.DataFrame(rows, columns=[k for k in statistics.keys()])

    rows2 = []
    for it in range(iterations):
        try:
            df2 = df[(df['iteration'] == it)]
            for col in columns:
                row = [it, col, df2[col].values.min(), df2[col].values.mean(), df2[col].values.std(), df2[col].values.max()]
                rows2.append(row)
        except Exception as ex:
            print(ex)

    df2 = pd.DataFrame(rows2, columns=['Iteration', 'Metric', 'Min', 'Avg', 'Std', 'Max'])

    df2.to_csv(file, index=False)

    return df2

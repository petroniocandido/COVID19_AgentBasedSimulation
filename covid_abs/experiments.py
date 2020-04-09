import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from covid_abs.common import *
from covid_abs.agents import *
from covid_abs.abs import *


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
    rows = []
    columns = None
    for experiment in range(experiments):
        sim = simulation_type(**kwargs)
        sim.initialize()
        if columns is None:
            statistics = sim.get_statistics(kind='all')
            columns = [k for k in statistics.keys()]
        for it in range(iterations):
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



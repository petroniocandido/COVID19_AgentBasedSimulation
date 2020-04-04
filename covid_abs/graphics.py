import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from covid_abs.common import *
from covid_abs.agents import *
from covid_abs.abs import *

from matplotlib import animation, rc
from IPython.display import HTML

legend_ecom = {'Q1': 'Most Poor', 'Q2': 'Poor', 'Q3': 'Working Class', 'Q4': 'Rich', 'Q5': 'Most Rich'}


def color1(s):
    if s == 'Susceptible':
        return 'lightblue'
    elif s == 'Infected':
        return 'gray'
    elif s == 'Recovered_Immune':
        return 'lightgreen'
    elif s == 'Death':
        return 'black'
    elif s == 'Hospitalization':
        return 'orange'
    elif s == 'Severe':
        return 'red'
    else:
        return 'white'


def color2(agent):
    if agent.status == Status.Susceptible:
        return 'lightblue'
    elif agent.status == Status.Infected:
        if agent.infected_status == InfectionSeverity.Asymptomatic:
            return 'gray'
        elif agent.infected_status == InfectionSeverity.Hospitalization:
            return 'orange'
        else:
            return 'red'
    elif agent.status == Status.Recovered_Immune:
        return 'lightgreen'
    elif agent.status == Status.Death:
        return 'black'


def color3(a):
    if a == 'Q1':
        return 'red'
    elif a == 'Q2':
        return 'orange'
    elif a == 'Q3':
        return 'yellow'
    elif a == 'Q4':
        return 'green'
    elif a == 'Q5':
        return 'blue'


def update_statistics(sim, statistics):
    stats1 = sim.get_statistics(kind='info')
    statistics['info'].append(stats1)
    df1 = pd.DataFrame(statistics['info'], columns=[k for k in stats1.keys()])

    stats2 = sim.get_statistics(kind='ecom')
    statistics['ecom'].append(stats2)
    df2 = pd.DataFrame(statistics['ecom'], columns=[k for k in stats2.keys()])

    return (df1, df2)


def clear(scat, linhas1, linhas2):
    for linha1 in linhas1.values():
        linha1.set_data([], [])

    for linha2 in linhas2.values():
        linha2.set_data([], [])

    ret = [scat]
    for l in linhas1.values():
        ret.append(l)
    for l in linhas2.values():
        ret.append(l)

    return tuple(ret)


def update(sim, scat, linhas1, linhas2, statistics):
    sim.execute()
    scat.set_facecolor([color2(a) for a in sim.get_population()])
    scat.set_offsets(sim.get_positions())

    df1, df2 = update_statistics(sim, statistics)

    for col in linhas1.keys():
        linhas1[col].set_data(df1.index.values, df1[col].values)

    for col in linhas2.keys():
        linhas2[col].set_data(df2.index.values, df2[col].values)

    ret = [scat]
    for l in linhas1.values():
        ret.append(l)
    for l in linhas2.values():
        ret.append(l)

    return tuple(ret)


def execute_simulation(sim, **kwargs):
    statistics = {'info': [], 'ecom': []}

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[20, 5])
    # plt.close()

    frames = kwargs.get('iterations', 100)
    iteration_time = kwargs.get('iteration_time', 250)

    sim.initialize()

    ax[0].set_title('Simulation Environment')
    ax[0].set_xlim((0, sim.length))
    ax[0].set_ylim((0, sim.height))

    pos = np.array(sim.get_positions())

    scat = ax[0].scatter(pos[:, 0], pos[:, 1],
                         c=[color2(a) for a in sim.get_population()])

    df1, df2 = update_statistics(sim, statistics)

    ax[1].set_title('Contagion Evolution')
    ax[1].set_xlim((0, frames))

    linhas1 = {}

    ax[1].axhline(y=sim.critical_limit, c="black", ls='--')

    for col in df1.columns.values:
        linhas1[col], = ax[1].plot(df1.index.values, df1[col].values, c=color1(col), label=col)

    ax[1].set_xlabel("Nº of Days")
    ax[1].set_ylabel("% of Population")

    handles, labels = ax[1].get_legend_handles_labels()
    lgd = ax[1].legend(handles, labels, loc=2, bbox_to_anchor=(0, 0))

    linhas2 = {}

    ax[2].set_title('Economical Impact')
    ax[2].set_xlim((0, frames))

    for col in df2.columns.values:
        linhas2[col], = ax[2].plot(df2.index.values, df2[col].values, c=color3(col), label=legend_ecom[col])

    ax[2].set_xlabel("Nº of Days")
    ax[2].set_ylabel("Wealth")

    handles, labels = ax[2].get_legend_handles_labels()
    lgd = ax[2].legend(handles, labels, loc=2, bbox_to_anchor=(1, 1))

    animate = lambda i: update(sim, scat, linhas1, linhas2, statistics)

    init = lambda: clear(scat, linhas1, linhas2)

    # animation function. This is called sequentially
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=iteration_time, blit=True)

    return anim


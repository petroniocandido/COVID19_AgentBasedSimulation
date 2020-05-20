"""
Auxiliary methods for plotting the simulations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from covid_abs.common import *
from covid_abs.agents import *
from covid_abs.abs import *

from matplotlib import animation, rc
from IPython.display import HTML

legend_ecom = {
    'Q1': 'Most Poor', 'Q2': 'Poor', 'Q3': 'Working Class',
    'Q4': 'Rich', 'Q5': 'Most Rich', 'Business':'Business', 'Government':'Government'
}
"""Legend for wealth distribution quintiles"""


def color1(s):
    """Plotting colors by status string"""
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
    """Plotting colors by Status and InfectionSeverity"""
    if agent.status == Status.Susceptible:
        return 'blue'
    elif agent.status == Status.Infected:
        if agent.infected_status == InfectionSeverity.Asymptomatic:
            return 'darkgray'
        elif agent.infected_status == InfectionSeverity.Hospitalization:
            return 'orange'
        else:
            return 'red'
    elif agent.status == Status.Recovered_Immune:
        return 'green'
    elif agent.status == Status.Death:
        return 'black'


def color3(a):
    """Plotting colors by wealth distribution quintiles"""
    if a == 'Q1':
        return 'red'
    elif a == 'Q2':
        return 'orange'
    elif a == 'Q3':
        return 'yellow'
    elif a == 'Q4':
        return 'blue'
    elif a == 'Q5':
        return 'purple'
    elif a == 'Business':
        return 'darkgreen'
    elif a == 'Government':
        return 'sienna'


def update_statistics(sim, statistics):
    """Store the iteration statistics"""

    stats1 = sim.get_statistics(kind='info')
    statistics['info'].append(stats1)
    df1 = pd.DataFrame(statistics['info'], columns=[k for k in stats1.keys()])

    stats2 = sim.get_statistics(kind='ecom')
    statistics['ecom'].append(stats2)
    df2 = pd.DataFrame(statistics['ecom'], columns=[k for k in stats2.keys()])

    return (df1, df2)


def clear(scat, linhas1, linhas2):
    """

    :param scat:
    :param linhas1:
    :param linhas2:
    :return:
    """
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
    """
    Execute an iteration of the simulation and update the animation graphics

    :param sim:
    :param scat:
    :param linhas1:
    :param linhas2:
    :param statistics:
    :return:
    """
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
    """
    Execute a simulation and plot its results

    :param sim: a Simulation or MultiopulationSimulation object
    :param iterations: number of interations of the simulation
    :param  iteration_time: time (in miliseconds) between each iteration
    :return: an animation object
    """
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

    ax[1].axhline(y=sim.critical_limit, c="black", ls='--', label='Critical limit')

    for col in df1.columns.values:
        if col != 'Asymptomatic':
            linhas1[col], = ax[1].plot(df1.index.values, df1[col].values, c=color1(col), label=col)

    ax[1].set_xlabel("Nº of Days")
    ax[1].set_ylabel("% of Population")

    handles, labels = ax[1].get_legend_handles_labels()
    lgd = ax[1].legend(handles, labels, loc='top right') #2, bbox_to_anchor=(0, 0))

    linhas2 = {}

    ax[2].set_title('Economical Impact')
    ax[2].set_xlim((0, frames))

    for col in df2.columns.values:
        linhas2[col], = ax[2].plot(df2.index.values, df2[col].values, c=color3(col), label=legend_ecom[col])

    ax[2].set_xlabel("Nº of Days")
    ax[2].set_ylabel("Wealth")

    handles, labels = ax[2].get_legend_handles_labels()
    lgd = ax[2].legend(handles, labels, loc='top right') #2, bbox_to_anchor=(1, 1))

    animate = lambda i: update(sim, scat, linhas1, linhas2, statistics)

    init = lambda: clear(scat, linhas1, linhas2)

    # animation function. This is called sequentially
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=iteration_time, blit=True)

    return anim


def clear_graph(ax, linhas1, linhas2):
    """

    :param scat:
    :param linhas1:
    :param linhas2:
    :return:
    """
    ax.clear()

    for linha1 in linhas1.values():
        linha1.set_data([], [])

    for linha2 in linhas2.values():
        linha2.set_data([], [])

    ret = []
    for l in linhas1.values():
        ret.append(l)
    for l in linhas2.values():
        ret.append(l)

    return tuple(ret)


def update_graph(sim, ax, linhas1, ax1, linhas2, ax2, statistics):
    """
    Execute an iteration of the simulation and update the animation graphics

    :param sim:
    :param scat:
    :param linhas1:
    :param linhas2:
    :param statistics:
    :return:
    """
    sim.execute()

    ax.clear()

    draw_graph(sim, ax=ax)

    df1, df2 = update_statistics(sim, statistics)

    for col in linhas1.keys():
        linhas1[col].set_data(df1.index.values, df1[col].values)

    ymax = 0
    for col in linhas2.keys():
        ymax = max(ymax, max(df2[col].values))
        linhas2[col].set_data(df2.index.values, df2[col].values)

    ax2.set_ylim(0, ymax)

    ret = []
    for l in linhas1.values():
        ret.append(l)
    for l in linhas2.values():
        ret.append(l)

    return tuple(ret)


def execute_graphsimulation(sim, **kwargs):
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                   AutoMinorLocator)

    statistics = {'info': [], 'ecom': []}

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[20, 5])
    # plt.close()

    frames = kwargs.get('iterations', 100)
    iteration_time = kwargs.get('iteration_time', 250)

    tick_unit = kwargs.get('tick_unit', 72)

    sim.initialize()

    ax[0].set_title('Simulation Environment')
    ax[0].set_xlim((0, sim.length))
    ax[0].set_ylim((0, sim.height))

    draw_graph(sim, ax=ax[0])

    df1, df2 = update_statistics(sim, statistics)

    tickslabels = [str(i//24) for i in range(0, frames, tick_unit)]

    #print(ticks)

    ax[1].set_title('Contagion Evolution')
    ax[1].set_xlim((0, frames))
    ax[1].set_ylim((0, 1))
    ax[1].xaxis.set_major_locator(MultipleLocator(tick_unit))
    ax[1].set_xticklabels(tickslabels)

    linhas1 = {}

    ax[1].axhline(y=sim.critical_limit, c="black", ls='--', label='Critical limit')

    for col in df1.columns.values:
        if col != 'Asymptomatic':
            linhas1[col], = ax[1].plot(df1.index.values, df1[col].values, c=color1(col), label=col)

    ax[1].set_xlabel("Nº of Days")
    ax[1].set_ylabel("% of Population")

    handles, labels = ax[1].get_legend_handles_labels()
    lgd = ax[1].legend(handles, labels, loc='top right')  # 2, bbox_to_anchor=(0, 0))

    linhas2 = {}

    ax[2].set_title('Economical Impact')
    ax[2].set_xlim((0, frames))
    ax[2].xaxis.set_major_locator(MultipleLocator(tick_unit))
    ax[2].set_xticklabels(tickslabels)

    #print(df2.columns.values)
    for col in df2.columns.values:
        linhas2[col], = ax[2].plot(df2.index.values, df2[col].values, c=color3(col), label=legend_ecom[col])

    ax[2].set_xlabel("Nº of Days")
    ax[2].set_ylabel("Wealth")

    handles, labels = ax[2].get_legend_handles_labels()
    lgd = ax[2].legend(handles, labels, loc='top right')  # 2, bbox_to_anchor=(1, 1))

    animate = lambda i: update_graph(sim, ax[0], linhas1, ax[1], linhas2, ax[2], statistics)

    init = lambda: clear_graph(ax[0], linhas1, linhas2)

    # animation function. This is called sequentially
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=iteration_time, blit=True,
                                   repeat=True)

    return anim


def draw_graph(sim, ax=None, edges=False):
    import networkx as nx
    from covid_abs.graphics import color2
    G = nx.Graph()
    colors = []
    pos = {}
    sizes = []

    G.add_node(sim.healthcare.id, type='healthcare')
    colors.append('darkseagreen')
    pos[sim.healthcare.id] = [sim.healthcare.x, sim.healthcare.y]
    sizes.append(100)

    for house in sim.houses:
        G.add_node(house.id, type='house')
        colors.append('cyan')
        pos[house.id] = [house.x, house.y]
        sizes.append(100)

    for bus in sim.business:
        G.add_node(bus.id, type='business')
        colors.append('darkviolet')
        pos[bus.id] = [bus.x, bus.y]
        sizes.append(100)

    for person in sim.population:
        G.add_node(person.id, type='person')
        colors.append(color2(person))
        pos[person.id] = [person.x, person.y]
        sizes.append(10)

    if edges:
        for house in sim.houses:
            for person in house.homemates:
                G.add_edge(house.id, person.id)

        for bus in sim.business:
            for person in bus.employees:
                G.add_edge(bus.id, person.id)

    nx.draw(G, ax=ax, pos=pos, node_color=colors, node_size=sizes)
    if ax is not None:
        ax.set_xlim((0, sim.length))
        ax.set_ylim((0, sim.height))


def draw_graph2(sim, ax=None, edges=False):
    import networkx as nx
    from covid_abs.graphics import color2
    G = nx.Graph()
    pos = {}

    G.add_node(sim.healthcare.id, type='healthcare')
    pos[sim.healthcare.id] = [sim.healthcare.x, sim.healthcare.y]

    houses = []
    for house in sim.houses:
        G.add_node(house.id, type='house')
        pos[house.id] = [house.x, house.y]
        houses.append(house.id)

    buss = []
    for bus in sim.business:
        G.add_node(bus.id, type='business')
        pos[bus.id] = [bus.x, bus.y]
        buss.append(bus.id)

    colors = {}
    for person in sim.population:
        G.add_node(person.id, type='person')
        col = color2(person)
        if col not in colors:
            colors[col] = {'status': person.status, 'severity': person.infected_status, 'id':[]}
        colors[col]['id'].append(person.id)
        pos[person.id] = [person.x, person.y]

    if edges:
        for house in sim.houses:
            for person in house.homemates:
                G.add_edge(house.id, person.id)

        for bus in sim.business:
            for person in bus.employees:
                G.add_edge(bus.id, person.id)

    #nx.draw(G, ax=ax, pos=pos, node_color=colors, node_size=sizes)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=[sim.healthcare.id], node_color='darkseagreen', label='Hospital')
    nx.draw_networkx_nodes(G, pos=pos, nodelist=houses, node_color='cyan', label='Houses')
    nx.draw_networkx_nodes(G, pos=pos, nodelist=buss, node_color='darkviolet', label='Business')
    for key in colors.keys():
        nx.draw_networkx_nodes(G, pos=pos, nodelist=[sim.healthcare.id], node_color='darkseagreen', label='Hospital')



def save_gif(anim, file, writer='imagemagick'):
    anim.save(file, writer='imagemagick', fps=60)



from covid_abs.graphics import *
from covid_abs.experiments import *
from covid_abs.network.graph_abs import *
from covid_abs.network.util import *
import numpy as np


def sleep(a):
    if not new_day(a.iteration) and bed_time(a.iteration):
        return True
    #elif 9 <= a.iteration % 24 <= 11 and 14 <= a.iteration % 24 <= 16:
    #    return True
    return False


def lockdown(a):
    if a.house is not None:
        a.house.checkin(a)
    return True


def conditional_lockdown(a):
    if a.environment.get_statistics()['Infected'] > .05:
        return lockdown(a)
    else:
        return False

isolated = []


def sample_isolated(environment, isolation_rate=.5, list_isolated=isolated):
    for a in environment.population:
        test = np.random.rand()
        if test <= isolation_rate:
            list_isolated.append(a.id)


def check_isolation(list_isolated, agent):
    if agent.id in list_isolated:
        agent.move_to_home()
        return True
    return False


def vertical_isolation(a):
  if a.economical_status == EconomicalStatus.Inactive:
    if a.house is not None:
      a.house.checkin(a)
    return True
  return False


global_parameters = dict(

    # General Parameters
    length=500,
    height=500,

    # Demographic
    population_size=300,
    homemates_avg=3,
    homeless_rate=0.0005,
    amplitudes={
        Status.Susceptible: 10,
        Status.Recovered_Immune: 10,
        Status.Infected: 10
    },

    # Epidemiological
    critical_limit=0.01,
    contagion_rate=.9,
    incubation_time=5,
    contagion_time=10,
    recovering_time=20,

    # Economical
    total_wealth=10000000,
    total_business=9,
    minimum_income=900.0,
    minimum_expense=600.0,
    public_gdp_share=0.1,
    business_gdp_share=0.5,
    unemployment_rate=0.12,
    business_distance=20
)

scenario0 = dict(
    name='scenario0',
    initial_infected_perc=.0,
    initial_immune_perc=1.,
    contagion_distance=1.,
    #callbacks={'on_execute': lambda x: sleep(x) }
)

scenario1 = dict(
    name='scenario1',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={'on_execute': lambda x: sleep(x) }
)

scenario2 = dict(
    name='scenario2',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_execute': lambda x: sleep(x),
        'on_person_move': lambda x: lockdown(x)
    }
)

scenario3 = dict(
    name='scenario3',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_execute': lambda x: sleep(x),
        'on_person_move': lambda x: conditional_lockdown(x)
    }
)

scenario4 = dict(
    name='scenario4',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_execute': lambda x: sleep(x),
        'on_person_move': lambda x: vertical_isolation(x)
    }
)

scenario5 = dict(
    name='scenario5',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_execute': lambda x: sleep(x),
        'post_initialize': lambda x: sample_isolated(x, isolation_rate=.4, list_isolated=isolated),
        'on_person_move': lambda x: check_isolation(isolated, x)
    }
)

scenario6 = dict(
    name='scenario6',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_execute': lambda x: sleep(x),
        'post_initialize': lambda x: sample_isolated(x, isolation_rate=.5, list_isolated=isolated),
        'on_person_move': lambda x: check_isolation(isolated, x)
    }
)

scenario7 = dict(
    name='scenario7',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_execute': lambda x: sleep(x),
        'post_initialize': lambda x: sample_isolated(x, isolation_rate=.6, list_isolated=isolated),
        'on_person_move': lambda x: check_isolation(isolated, x)
    }
)


def pset(x, property, value):
    x.__dict__[property] = value
    return False


scenario8 = dict(
    name='scenario8',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=.05,
    callbacks={
        'on_execute': lambda x: sleep(x) ,
        'on_initialize': lambda x: pset(x, 'contagion_rate', 0.1)
    }
)

scenario9 = dict(
    name='scenario9',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=.05,
    callbacks={
        'on_execute': lambda x: sleep(x) ,
        'post_initialize': lambda x: sample_isolated(x, isolation_rate=.6, list_isolated=isolated),
        'on_person_move': lambda x: check_isolation(isolated, x),
        'on_initialize': lambda x: pset(x, 'contagion_rate', 0.1)
    }
)

scenario10 = dict(
    name='scenario10',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=.05,
    callbacks={
        'on_execute': lambda x: sleep(x) ,
        'post_initialize': lambda x: sample_isolated(x, isolation_rate=.5, list_isolated=isolated),
        'on_person_move': lambda x: check_isolation(isolated, x),
        'on_initialize': lambda x: pset(x, 'contagion_rate', 0.1)
    }
)


def reset(lista):
    lista = []


part_isol0 = dict(
    name='partialisolation0.0',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        #'on_initialize': lambda x: reset(isolated),
        'on_execute': lambda x: sleep(x),
        #'post_initialize': lambda x: sample_isolated(x, isolation_rate=0.1, list_isolated=isolated),
        #'on_person_move': lambda x: check_isolation(isolated, x)
    }
)

part_isol11 = dict(
    name='partialisolation1.0',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_initialize': lambda x: reset(isolated),
        'on_execute': lambda x: sleep(x),
        'post_initialize': lambda x: sample_isolated(x, isolation_rate=1., list_isolated=isolated),
        'on_person_move': lambda x: check_isolation(isolated, x)
    }
)

part_isol1 = dict(
    name='partialisolation0.1',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_initialize': lambda x: reset(isolated),
        'on_execute': lambda x: sleep(x),
        'post_initialize': lambda x: sample_isolated(x, isolation_rate=0.1, list_isolated=isolated),
        'on_person_move': lambda x: check_isolation(isolated, x)
    }
)


part_isol2 = dict(
    name='partialisolation0.2',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_initialize': lambda x: reset(isolated),
        'on_execute': lambda x: sleep(x),
        'post_initialize': lambda x: sample_isolated(x, isolation_rate=0.2, list_isolated=isolated),
        'on_person_move': lambda x: check_isolation(isolated, x)
    }
)


part_isol3 = dict(
    name='partialisolation0.3',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_initialize': lambda x: reset(isolated),
        'on_execute': lambda x: sleep(x),
        'post_initialize': lambda x: sample_isolated(x, isolation_rate=0.3, list_isolated=isolated),
        'on_person_move': lambda x: check_isolation(isolated, x)
    }
)

part_isol4 = dict(
    name='partialisolation0.4',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_initialize': lambda x: reset(isolated),
        'on_execute': lambda x: sleep(x),
        'post_initialize': lambda x: sample_isolated(x, isolation_rate=0.4, list_isolated=isolated),
        'on_person_move': lambda x: check_isolation(isolated, x)
    }
)


part_isol5 = dict(
    name='partialisolation0.5',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_initialize': lambda x: reset(isolated),
        'on_execute': lambda x: sleep(x),
        'post_initialize': lambda x: sample_isolated(x, isolation_rate=0.5, list_isolated=isolated),
        'on_person_move': lambda x: check_isolation(isolated, x)
    }
)


part_isol6 = dict(
    name='partialisolation0.6',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_initialize': lambda x: reset(isolated),
        'on_execute': lambda x: sleep(x),
        'post_initialize': lambda x: sample_isolated(x, isolation_rate=0.6, list_isolated=isolated),
        'on_person_move': lambda x: check_isolation(isolated, x)
    }
)


part_isol7 = dict(
    name='partialisolation0.7',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_initialize': lambda x: reset(isolated),
        'on_execute': lambda x: sleep(x),
        'post_initialize': lambda x: sample_isolated(x, isolation_rate=0.7, list_isolated=isolated),
        'on_person_move': lambda x: check_isolation(isolated, x)
    }
)


part_isol8 = dict(
    name='partialisolation0.8',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_initialize': lambda x: reset(isolated),
        'on_execute': lambda x: sleep(x),
        'post_initialize': lambda x: sample_isolated(x, isolation_rate=0.8, list_isolated=isolated),
        'on_person_move': lambda x: check_isolation(isolated, x)
    }
)


part_isol9 = dict(
    name='partialisolation0.9',
    initial_infected_perc=.01,
    initial_immune_perc=.01,
    contagion_distance=1.,
    callbacks={
        'on_initialize': lambda x: reset(isolated),
        'on_execute': lambda x: sleep(x),
        'post_initialize': lambda x: sample_isolated(x, isolation_rate=0.9, list_isolated=isolated),
        'on_person_move': lambda x: check_isolation(isolated, x)
    }
)


#for scenario in scenario0, scenario1, scenario2, scenario3, scenario4, scenario5, scenario6, scenario7, scenario8, scenario9]:
for part_isol in [part_isol0, part_isol11]:
    '''
    sim = GraphSimulation(
        **{**global_parameters, **scenario}
    )
    anim = execute_graphsimulation(sim, iterations=1440, iteration_time=25)

    anim.save("{}_new.mp4".format(scenario['name']), writer='ffmpeg', fps=60)
    '''
    #print(scenario['name'])

    isolated = []
    batch_experiment(5, 1440, "{}.csv".format(part_isol['name']),
                     simulation_type=GraphSimulation,
                     verbose='experiments',
                     **{**global_parameters, **part_isol}
                     )

'''
sim.append_trigger_simulation(
    lambda s: s.get_statistics()['Infected'] > 0.1,
    'amplitudes',
    lambda s: {
        Status.Susceptible: 0.1,
        Status.Recovered_Immune: 0.1,
        Status.Infected: 0.1
    }
)

sim.append_trigger_simulation(
    lambda s: s.get_statistics()['Infected'] > 0.1,
    'execute',
    lambda s: s.apply_business('open', True, 'open', False)
)

sim.append_trigger_simulation(
    lambda s: s.get_statistics()['Infected'] < 0.1 and s.get_statistics()['Recovered_Immune'] > 0.05,
    'amplitudes',
    lambda s: {
        Status.Susceptible: 10,
        Status.Recovered_Immune: 10,
        Status.Infected: 10
    }
)

sim.append_trigger_simulation(
    lambda s: s.get_statistics()['Infected'] < 0.1 and s.get_statistics()['Recovered_Immune'] > 0.05,
    'execute',
    lambda s: s.apply_business('open', False, 'open', True)
)
'''

'''
sim.append_trigger_simulation(
    lambda s: s.get_statistics()['Infected'] > 0.1,
    'execute',
    lambda s: s.append_trigger_population(lambda x, p: True,
                              'move',
                              lambda a, p: mov_check(a, a.house) )
)

sim.append_trigger_simulation(
    lambda s: s.get_statistics()['Infected'] < 0.1 and s.get_statistics()['Recovered_Immune'] > 0.05,
    'execute',
    lambda s: s.remove_trigger_population('move')
)

'''
#'''


# sim.apply_business('open', True, 'open', False)


def mov_check(a, b):
    if b is not None:
        b.checkin(a)
        return b.x + np.random.normal(0, 0.01, 1), b.y + np.random.normal(0, 0.01, 1)
    return a.x + np.random.normal(0, 0.01, 1), a.y + np.random.normal(0, 0.01, 1)


#'''
#sim.append_trigger_population(lambda x: True, 'move_freely', lambda a: (a.x, a.y) )
#sim.append_trigger_population(lambda x: not x.is_unemployed(), 'move_work', lambda a: (a.x, a.y) )
#sim.append_trigger_population(lambda x,s: x.status == Status.Infected and x.infected_status == InfectionSeverity.Asymptomatic,
#                              'move', lambda a,s: mov_check(a, a.house) )

#sim.append_trigger_population(lambda x, s: True, 'move', lambda a, s: mov_check(a, a.house))


# anim = execute_graphsimulation(sim, iterations=1440, iteration_time=25)

# save_gif(anim, teste.mp4)

# plt.plot()


'''
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.show()
sim.initialize()
sim.iteration = 719
for i in range(1440):
    #print(i%24, day_of_week(i), day_of_month(i), work_day(i))
    sim.execute()
    #for bus in sim.business:
    #    print(bus.num_employees, bus.wealth, bus.social_stratum, bus.price, bus.sales)
    #break
    #plt.clf()
    #sleep(1)
    if i%24 == 0:
        print("{}".format(sim.get_statistics('all')))
    #print("{}".format(sim.get_statistics('ecom')))
    #ax.clear()
    #plt.set_title("{}".format(i))
    #
    draw_graph(sim) #, ax=ax)
    plt.show()
'''

from covid_abs.experiments import batch_experiment, plot_graph_batch_results

'''
tmp = batch_experiment(35, 1440, "scenario2.csv",
                       simulation_type=GraphSimulation,
                       verbose='experiments',
                       # Percentage of infected in initial population
                       initial_infected_perc=.01,
                       # Percentage of immune in initial population
                       initial_immune_perc=.01,
                       # Length of simulation environment
                       length=500,
                       # Height of simulation environment
                       height=500,
                       # Size of population
                       population_size=100,
                       # Minimal distance between agents for contagion
                       contagion_distance=1.,
                       contagion_rate=.9,
                       # Maximum percentage of population which Healthcare System can handle simutaneously
                       critical_limit=0.05,
                       # Mobility ranges for agents, by Status
                       amplitudes={
                           Status.Susceptible: 10,
                           Status.Recovered_Immune: 10,
                           Status.Infected: 1
                       },
                       total_wealth=10000000,
                       total_business=10,
                       minimum_income=900.0,
                       minimum_expense=650.0,
                       population_move_triggers=[
                           {'condition': lambda a: a.status == Status.Infected,
                            'attribute': 'move',
                            'action': lambda a, s: (s.healthcare.x + np.random.normal(0, 0.01, 1),
                                                    s.healthcare.y + np.random.normal(0, 0.01, 1))
                            }
                       ]
                       )
'''
'''
df = pd.read_csv('scenario2.csv')
plot_graph_batch_results(df)
'''
print("")

"""
df = pd.read_csv('scenario1.csv')

plot_batch_results(df)

plt.show()


batch_experiment(50, 80, "scenario1.csv",
                 # Percentage of infected in initial population
                 initial_infected_perc=0.02,
                 # Percentage of immune in initial population
                 initial_immune_perc=0.01,
                 # Length of simulation environment
                 length=100,
                 # Height of simulation environment
                 height=100,
                 # Size of population
                 population_size=80,
                 # Minimal distance between agents for contagion
                 contagion_distance=5.,
                 # Maximum percentage of population which Healthcare System can handle simutaneously
                 critical_limit=0.05,
                 # Mobility ranges for agents, by Status
                 amplitudes={
                     Status.Susceptible: 5,
                     Status.Recovered_Immune: 5,
                     Status.Infected: 5
                 }
                 )

batch_experiment(50, 80, "scenario2.csv",
                 # Percentage of infected in initial population
                 initial_infected_perc=0.02,
                 # Percentage of immune in initial population
                 initial_immune_perc=0.01,
                 # Length of simulation environment
                 length=100,
                 # Height of simulation environment
                 height=100,
                 # Size of population
                 population_size=80,
                 # Minimal distance between agents for contagion
                 contagion_distance=5.,
                 # Maximum percentage of population which Healthcare System can handle simutaneously
                 critical_limit=0.05,
                 # Mobility ranges for agents, by Status
                 amplitudes={
                     Status.Susceptible: 5,
                     Status.Recovered_Immune: 5,
                     Status.Infected: 0
                 }
                 )

batch_experiment(50, 80, "scenario3.csv",
                 # Percentage of infected in initial population
                 initial_infected_perc=0.02,
                 # Percentage of immune in initial population
                 initial_immune_perc=0.01,
                 # Length of simulation environment
                 length=100,
                 # Height of simulation environment
                 height=100,
                 # Size of population
                 population_size=80,
                 # Minimal distance between agents for contagion
                 contagion_distance=5.,
                 # Maximum percentage of population which Healthcare System can handle simutaneously
                 critical_limit=0.05,
                 # Mobility ranges for agents, by Status
                 amplitudes={
                     Status.Susceptible: 0.5,
                     Status.Recovered_Immune: 0.5,
                     Status.Infected: 0
                 }
                 )

batch_experiment(50, 120, "scenario4.csv",
                 length=100,
                 height=100,
                 initial_infected_perc=0.02,
                 population_size=80,
                 contagion_distance=5,
                 critical_limit=.1,
                 amplitudes={
                     Status.Susceptible: 5,
                     Status.Recovered_Immune: 5,
                     Status.Infected: 5
                 },
                 triggers_simulation=[
                     {'condition': lambda a: a.get_statistics()['Infected'] >= .2,
                      'attribute': 'amplitudes',
                      'action': lambda a: {
                          Status.Susceptible: 1.5,
                          Status.Recovered_Immune: 1.5,
                          Status.Infected: 1.5
                      }},
                     {'condition': lambda a: a.get_statistics()['Infected'] <= .05,
                      'attribute': 'amplitudes',
                      'action': lambda a: {
                          Status.Susceptible: 5,
                          Status.Recovered_Immune: 5,
                          Status.Infected: 5
                      }}
                 ])


sim1 = Simulation(initial_infected_perc=0.02,
                  length=100,
                  height=100,
                  population_size=80,
                  contagion_distance=5.,
                  critical_limit=.1,
                  amplitudes={
                      Status.Susceptible: 5,
                      Status.Recovered_Immune: 5,
                      Status.Infected: 5
                  })

sim2 = Simulation(initial_infected_perc=0.02,
                  length=100,
                  height=100,
                  population_size=80,
                  contagion_distance=5.,
                  critical_limit=.1,
                  amplitudes={
                      Status.Susceptible: 1,
                      Status.Recovered_Immune: 1,
                      Status.Infected: 1
                  })

batch_experiment(50, 80, "scenario5.csv", simulation_type=MultiPopulationSimulation,
                 length=200,
                 height=200,
                 contagion_distance=5.,
                 critical_limit=.1,
                 simulations=[sim1, sim2],
                 positions=[(0, 0), (80, 80)],
                 total_population=160
                 )
"""

from covid_abs.graphics import  *
from covid_abs.network.graph_abs import *
from covid_abs.network.util import *



'''
sim = GraphSimulation(  # Percentage of infected in initial population
    initial_infected_perc=0.01,
    # Percentage of immune in initial population
    initial_immune_perc=.0,
    # Length of simulation environment
    length=500,
    # Height of simulation environment
    height=500,
    # Size of population
    population_size=100,
    # Minimal distance between agents for contagion
    contagion_distance=.1,
    contagion_rate=.3,
    # Maximum percentage of population which Healthcare System can handle simutaneously
    critical_limit=0.05,
    # Mobility ranges for agents, by Status
    amplitudes={
        Status.Susceptible: 10,
        Status.Recovered_Immune: 10,
        Status.Infected: 10
    },
    total_wealth=10000000,
    total_business=10,
    minimum_income=900.0,
    minimum_expense=650.0
)
'''
#sim.apply_business('open', True, 'open', False)


def mov_check(a, b):
    if b is not None:
        b.checkin(a)
    return a.x, a.y

'''
sim.append_trigger_population(lambda x: True, 'move_freely', lambda a: (a.x, a.y) )
sim.append_trigger_population(lambda x: not x.is_unemployed(), 'move_work', lambda a: (a.x, a.y) )
sim.append_trigger_population(lambda x: True, 'move_home', lambda a: mov_check(a, a.house) )
'''

#anim = execute_graphsimulation(sim, iterations=1440, iteration_time=25)
#anim = execute_graphsimulation(sim, iterations=240, iteration_time=25)

#anim.save("scenario2.mp4", writer='ffmpeg', fps=60)

#save_gif(anim, teste.mp4)

#plt.plot()


'''
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.show()
sim.initialize()
for i in range(1440):
    #print(i%24, day_of_week(i), day_of_month(i), work_day(i))
    sim.execute()
    #for bus in sim.business:
    #    print(bus.num_employees, bus.wealth, bus.social_stratum, bus.price, bus.sales)
    #break
    #plt.clf()
    #sleep(1)
    if i%24 == 0:
        print("{}".format(sim.get_statistics('ecom')))
    #print("{}".format(sim.get_statistics('ecom')))
    #ax.clear()
    #plt.set_title("{}".format(i))
    #
    #draw_graph(sim) #, ax=ax)
    #plt.show()
'''

from covid_abs.experiments import batch_experiment

batch_experiment(50, 1440, "scenario1.csv",
                 simulation_type=GraphSimulation,
                 verbose='experiments',
                 # Percentage of infected in initial population
    initial_infected_perc=0.0,
    # Percentage of immune in initial population
    initial_immune_perc=1.0,
    # Length of simulation environment
    length=500,
    # Height of simulation environment
    height=500,
    # Size of population
    population_size=100,
    # Minimal distance between agents for contagion
    contagion_distance=.1,
    contagion_rate=.9,
    # Maximum percentage of population which Healthcare System can handle simutaneously
    critical_limit=0.05,
    # Mobility ranges for agents, by Status
    amplitudes={
        Status.Susceptible: 10,
        Status.Recovered_Immune: 10,
        Status.Infected: 10
    },
    total_wealth=10000000,
    total_business=10,
    minimum_income=900.0,
    minimum_expense=650.0
  )
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

from covid_abs.agents import *
from covid_abs.abs import *
from covid_abs.common import *
from covid_abs.experiments import *

batch_experiment(10, 80, "scenario1.csv",
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

batch_experiment(10, 80, "scenario2.csv",
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

batch_experiment(10, 80, "scenario3.csv",
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

batch_experiment(10, 80, "scenario4.csv",
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
                 trigger_simulation=[
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

batch_experiment(10, 80, "scenario5.csv", simulation_type=MultiPopulationSimulation,
                 length=200,
                 height=200,
                 contagion_distance=5.,
                 critical_limit=.1,
                 simulations=[sim1, sim2],
                 positions=[(0, 0), (80, 80)],
                 total_population=160
                 )

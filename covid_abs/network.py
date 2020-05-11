"""
Graph induced
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from covid_abs.common import *
from covid_abs.agents import *
from covid_abs.abs import *


def check_time(iteration, start, end):
    return start < iteration % 24 < end


def bed_time(iteration):
    return check_time(iteration,0,8)


def work_time(iteration):
    return check_time(iteration,8,16)


def free_time(iteration):
    return check_time(iteration,16,24)


class EconomicalStatus(Enum):
    Active = 1
    Inactive = 0


class Business(Agent):
    """
    The container of Agent's attributes and status
    """
    def __init__(self, **kwargs):
        super(Business, self).__init__(**kwargs)
        self.employees = []
        self.type = AgentType.Business
        self.incomes = 0
        self.expenses = 0
        self.price = kwargs.get("price",1.0)
        self.stocks = 10
        self.open = True

    def append_employee(self, agent):
        self.employees.append(agent)
        self.expenses += agent.income
        agent.employer = self

    def demand(self, agent):
        """Expenses due to employee payments"""
        agent.supply(agent.income)
        self.wealth -= agent.income

    def supply(self, agent):
        """Incomes due to selling product/service"""
        agent.demand(self.price)
        self.wealth += self.price
        self.stocks -= 1


class House(Agent):
    """
    The container of Agent's attributes and status
    """

    def __init__(self, **kwargs):
        super(House, self).__init__(**kwargs)
        self.homemates = []
        self.type = AgentType.House
        self.size = 0
        self.incomes = 0
        self.expenses = 0

    def append_mate(self, agent):
        self.homemates.append(agent)
        self.wealth += agent.wealth
        self.size += 1
        agent.house = self
        agent.x = int(self.x + np.random.normal(0.0, 0.5, 1))
        agent.y = int(self.y + np.random.normal(0.0, 0.5, 1))

    def demand(self, value = 0.0):
        """Expense of consuming product/services"""
        self.wealth -= value
        self.expenses += value

    def supply(self, value = 0.0):
        """Income of work"""
        self.wealth += value
        self.incomes += value


class Person(Agent):
    """
    The container of Agent's attributes and status
    """

    def __init__(self, **kwargs):
        super(Person, self).__init__(**kwargs)
        self.employer = kwargs.get("employer", None)
        self.house = kwargs.get("house", None)
        self.type = AgentType.Person
        self.economical_status = EconomicalStatus.Inactive
        self.incomes = kwargs.get("income", 0.0)
        self.expenses = kwargs.get("expense", 0.0)

        if 18 <= self.age <= 65:
            self.economical_status = EconomicalStatus.Active


    def demand(self, value = 0.0):
        """Expense for product/services"""
        self.house.demand(value)

    def supply(self, value = 0.0):
        """Income for work"""
        self.house.supply(value)

    def move_to_work(self):
        if self.economical_status == EconomicalStatus.Active:
            if self.employer is not None and self.employer.open:
                self.x = int(self.employer.x + np.random.normal(0.0, 0.5, 1))
                self.y = int(self.employer.y + np.random.normal(0.0, 0.5, 1))
            elif self.employer is None:
                self.x = int(self.x + np.random.normal(0.0, 1, 1))
                self.y = int(self.y + np.random.normal(0.0, 1, 1))

    def move_to_home(self):
        if self.house is not None:
            if self.employer is not None and self.employer.open:
                self.x = int(self.house.x + np.random.normal(0.0, 0.5, 1))
                self.y = int(self.house.y + np.random.normal(0.0, 0.5, 1))
            else:
                self.x = int(self.x + np.random.normal(0.0, 1, 1))
                self.y = int(self.y + np.random.normal(0.0, 1, 1))

    def move_freely(self):
        self.x = int(self.x + np.random.normal(0.0, 1, 1))
        self.y = int(self.y + np.random.normal(0.0, 1, 1))


class GraphSimulation(Simulation):
    def __init__(self, **kwargs):
        super(GraphSimulation, self).__init__(**kwargs)
        self.total_population = kwargs.get('total_population', 0)
        self.total_business  = kwargs.get('total_business', 10)
        self.government = None
        self.business = []
        self.houses = []
        self.homeless_rate = kwargs.get("homeless_rate", 0.001)
        self.unemployment_rate = kwargs.get("unemployment_rate", 0.1)
        self.homemates_avg = kwargs.get("homemates_avg", 5)
        self.homemates_std = kwargs.get("homemates_std", 1)
        self.iteration = -1

    def create_business(self):
        x, y = self.random_position()
        social_stratum = int(np.random.rand(1) * 100 // 20)
        self.business.append(Business(x=x, y=y, status=Status.Susceptible, social_stratum=social_stratum))

    def create_house(self):
        x, y = self.random_position()
        social_stratum = int(np.random.rand(1) * 100 // 20)
        self.houses.append(House(x=x, y=y, status=Status.Susceptible, social_stratum=social_stratum))

    def create_agent(self, status):
        """
        Create a new agent with the given status

        :param status: a value of agents.Status enum
        :return: the newly created agent
        """
        #x, y = self.random_position()

        age = int(np.random.beta(2, 5, 1) * 100)
        social_stratum = int(np.random.rand(1) * 100 // 20)
        self.population.append(Person(age=age, status=status, social_stratum=social_stratum))

    def initialize(self):
        """
        Initializate the Simulation by creating its population of agents
        """


        #number of houses
        for i in np.arange(0, int(self.population_size // self.homemates_avg)):
            self.create_house()

        # number of business
        for i in np.arange(0, self.total_business):
            self.create_business()

        # Initial infected population
        for i in np.arange(0, int(self.population_size * self.initial_infected_perc)):
            self.create_agent(Status.Infected)

        # Initial immune population
        for i in np.arange(0, int(self.population_size * self.initial_immune_perc)):
            self.create_agent(Status.Recovered_Immune)

        # Initial susceptible population
        for i in np.arange(0, self.population_size - len(self.population)):
            self.create_agent(Status.Susceptible)


        # Share the common wealth of 10^4 among the population, according each agent social stratum

        for quintile in [0, 1, 2, 3, 4]:

            houses = [x for x in filter(lambda x: x.social_stratum == quintile, self.houses)]
            nhouses = len(houses)

            total = lorenz_curve[quintile] * self.total_wealth
            qty = max(1.0, np.sum([1 for a in self.population if
                                   a.social_stratum == quintile and a.economical_status == EconomicalStatus.Active]))
            ag_share = total / qty

            for agent in filter(lambda x: x.social_stratum == quintile, self.population):

                # distribute wealth

                if agent.economical_status == EconomicalStatus.Active:
                    agent.wealth = ag_share
                    agent.income = basic_income[agent.social_stratum] * self.minimum_expense

                    # distribute employ
                    ix = np.random.randint(0, self.total_business)
                    self.business[ix].append_employee(agent)

                agent.expenses = self.minimum_expense/30

                #distribute habitation

                ix = np.random.randint(0, nhouses)
                house = houses[ix]
                house.append_mate(agent)

    def execute(self):
        self.iteration += 1

        bed = bed_time(self.iteration)
        work = work_time(self.iteration)
        free = free_time(self.iteration)

        for agent in filter(lambda x: x.status != Status.Death, self.population):
            if bed:
                agent.move_to_home()
            elif work:
                agent.move_to_work()
            else:
                agent.move_freely()

            self.update(agent)

        contacts = []

        for i in np.arange(0, self.population_size):
            for j in np.arange(i + 1, self.population_size):
                ai = self.population[i]
                aj = self.population[j]
                if ai.status == Status.Death or ai.status == Status.Death:
                    continue

                if np.sqrt((ai.x - aj.x) ** 2 + (ai.y - aj.y) ** 2) <= self.contagion_distance:
                    contacts.append((i, j))

        for par in contacts:
            ai = self.population[par[0]]
            aj = self.population[par[1]]
            self.contact(ai, aj)
            self.contact(aj, ai)

        self.statistics = None

    def update(self, agent):
        """
        Update the status of the agent

        :param agent: an instance of agents.Agent
        """

        if agent.status == Status.Death:
            return

        if agent.status == Status.Infected:
            agent.infected_time += 1/24

            indice = agent.age // 10 - 1 if agent.age > 10 else 0

            teste_sub = np.random.random()

            if agent.infected_status == InfectionSeverity.Asymptomatic:
                if age_hospitalization_probs[indice] > teste_sub:
                    agent.infected_status = InfectionSeverity.Hospitalization
            elif agent.infected_status == InfectionSeverity.Hospitalization:
                if age_severe_probs[indice] > teste_sub:
                    agent.infected_status = InfectionSeverity.Severe
                    self.get_statistics()
                    if self.statistics['Severe'] + self.statistics['Hospitalization'] >= self.critical_limit:
                        agent.status = Status.Death
                        agent.infected_status = InfectionSeverity.Asymptomatic

            death_test = np.random.random()
            if age_death_probs[indice] > death_test:
                agent.status = Status.Death
                agent.infected_status = InfectionSeverity.Asymptomatic
                return

            if agent.infected_time > 20:
                agent.infected_time = 0
                agent.status = Status.Recovered_Immune
                agent.infected_status = InfectionSeverity.Asymptomatic


def draw(sim, ax=None):
    import networkx as nx
    from covid_abs.graphics import color2
    G = nx.Graph()
    colors = []
    pos = {}
    for person in sim.population:
        G.add_node(person.id, type='person')
        colors.append(color2(person))
        pos[person.id] = [person.x, person.y]

    for house in sim.houses:
        G.add_node(house.id, type='house')
        colors.append('darkblue')
        pos[house.id] = [house.x, house.y]
        for person in house.homemates:
            G.add_edge(house.id, person.id)

    for bus in sim.business:
        G.add_node(bus.id, type='business')
        colors.append('darkred')
        pos[bus.id] = [bus.x, bus.y]
        for person in bus.employees:
            G.add_edge(bus.id, person.id)

    #nx.draw(G, node_color=colors)
    nx.draw(G, ax=ax, pos=pos, node_color=colors)


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
    return start <= iteration % 24 < end


def new_day(iteration):
    return iteration % 24 == 0


def day_of_week(iteration):
    return (iteration // 24) % 7 + 1


def work_day(iteration):
    wd = day_of_week(iteration)
    return wd not in [1, 7]


def day_of_month(iteration):
    return (iteration // 24) % 30 + 1


def new_month(iteration):
    return day_of_month(iteration) == 1 and iteration % 24 == 0


def bed_time(iteration):
    return check_time(iteration,0,8)


def work_time(iteration):
    return check_time(iteration,8,16)


def lunch_time(iteration):
    return iteration % 24 == 12


def free_time(iteration):
    return check_time(iteration,17,24)


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
        self.num_employees = 0
        self.type = AgentType.Business
        self.incomes = 0
        self.expenses = kwargs.get('expenses')
        self.labor_expenses = {}
        self.price = kwargs.get("price", np.max([1.0, np.random.normal(5, 1)]))
        self.stocks = 10
        self.sales = 0
        self.open = True

    def cash(self, value):
        self.wealth += value

    def hire(self, agent):
        self.employees.append(agent)
        self.labor_expenses[agent.id] = 0.0
        agent.employer = self
        self.num_employees += 1

    def fire(self, agent):
        self.employees.remove(agent)
        self.labor_expenses[agent.id] = None
        agent.employer = None
        self.num_employees -= 1

    def demand(self, agent):
        """Expenses due to employee payments"""
        agent.supply(self.labor_expenses[agent.id])
        self.cash(-self.labor_expenses[agent.id])
        self.labor_expenses[agent.id] = 0

    def supply(self, agent):
        """Incomes due to selling product/service"""
        qty = np.random.randint(1, 10)
        if qty > self.stocks:
            qty = self.stocks
        value = self.price * agent.social_stratum * qty
        agent.demand(value)
        self.cash(value)
        self.incomes += value
        self.stocks -= qty
        self.sales += qty

    def checkin(self, agent):
        """Employee is working"""
        self.stocks += 1
        self.labor_expenses[agent.id] += agent.income / 160

    def taxes(self, government):
        """Expenses due to employee payments"""
        tax = government.price * self.num_employees + government.price * self.social_stratum
        government.cash(tax)
        self.cash(-tax)

    def accounting(self, government):
        for person in self.employees:
            self.demand(person)
        self.taxes(government)
        self.incomes = 0
        self.sales = 0


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

    def checkin(self, agent):
        self.demand(agent.expenses/720)

    def demand(self, value = 0.0):
        """Expense of consuming product/services"""
        self.wealth -= value
        self.expenses += value

    def supply(self, value = 0.0):
        """Income of work"""
        self.wealth += value
        self.incomes += value

    def accounting(self, government):
        """Expenses due to employee payments"""
        tax = government.price * len(self.homemates) + government.price * self.social_stratum
        government.cash(tax)
        self.wealth -= tax
        self.incomes = 0
        self.expenses = 0


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
        if self.infected_status != InfectionSeverity.Asymptomatic:
            return

        if self.economical_status == EconomicalStatus.Active:
            if self.employer is not None and self.employer.open:
                self.x = int(self.employer.x + np.random.normal(0.0, 0.25, 1))
                self.y = int(self.employer.y + np.random.normal(0.0, 0.25, 1))
                self.employer.checkin(self)
            elif self.employer is None:
                self.x = int(self.x + np.random.normal(0.0, 1, 1))
                self.y = int(self.y + np.random.normal(0.0, 1, 1))

    def move_to_home(self):
        if self.infected_status != InfectionSeverity.Asymptomatic:
            return

        if self.house is not None:
            self.house.checkin(self)
            self.x = int(self.house.x + np.random.normal(0.0, 0.25, 1))
            self.y = int(self.house.y + np.random.normal(0.0, 0.25, 1))
        else:
            self.wealth -= self.incomes / 720
            self.x = int(self.x + np.random.normal(0.0, 1, 1))
            self.y = int(self.y + np.random.normal(0.0, 1, 1))

    def move_freely(self, range):
        if self.infected_status != InfectionSeverity.Asymptomatic:
            return

        self.x = int(self.x + np.random.normal(0, range, 1))
        self.y = int(self.y + np.random.normal(0, range, 1))

    def move_to(self, agent):
        self.x = int(agent.x + np.random.normal(0.0, 0.25, 1))
        self.y = int(agent.y + np.random.normal(0.0, 0.25, 1))


class GraphSimulation(Simulation):
    def __init__(self, **kwargs):
        super(GraphSimulation, self).__init__(**kwargs)
        self.total_population = kwargs.get('total_population', 0)
        self.total_business  = kwargs.get('total_business', 10)
        self.government = None
        self.business = []
        self.houses = []
        self.healthcare = None
        self.homeless_rate = kwargs.get("homeless_rate", 0.001)
        self.unemployment_rate = kwargs.get("unemployment_rate", 0.1)
        self.homemates_avg = kwargs.get("homemates_avg", 5)
        self.homemates_std = kwargs.get("homemates_std", 1)
        self.iteration = -1

    def create_business(self, social_stratum=None):
        x, y = self.random_position()
        if social_stratum is None:
            social_stratum = int(np.random.rand(1) * 100 // 20)
        self.business.append(Business(x=x, y=y, status=Status.Susceptible, social_stratum=social_stratum,
                                      expenses=self.government.price))

    def create_house(self, social_stratum=None):
        x, y = self.random_position()
        if social_stratum is None:
            social_stratum = int(np.random.rand(1) * 100 // 20)
        self.houses.append(House(x=x, y=y, status=Status.Susceptible, social_stratum=social_stratum))

    def create_agent(self, status, social_stratum=None):
        """
        Create a new agent with the given status

        :param status: a value of agents.Status enum
        :return: the newly created agent
        """
        #x, y = self.random_position()

        age = int(np.random.beta(2, 5, 1) * 100)
        if social_stratum is None:
            social_stratum = int(np.random.rand(1) * 100 // 20)
        self.population.append(Person(age=age, status=status, social_stratum=social_stratum))

    def initialize(self):
        """
        Initializate the Simulation by creating its population of agents
        """
        x, y = self.random_position()
        self.healthcare = Business(x=x, y=y, status=Status.Susceptible)
        x, y = self.random_position()
        self.government = Business(x=x, y=y, status=Status.Susceptible)

        #number of houses
        for i in np.arange(0, int(self.population_size // self.homemates_avg)):
            self.create_house(social_stratum=i % 5)

        # number of business
        for i in np.arange(0, self.total_business):
            self.create_business(social_stratum=i % 5)

        # Initial infected population
        for i in np.arange(0, int(self.population_size * self.initial_infected_perc)):
            self.create_agent(Status.Infected)

        # Initial immune population
        for i in np.arange(0, int(self.population_size * self.initial_immune_perc)):
            self.create_agent(Status.Recovered_Immune)

        # Initial susceptible population
        for i in np.arange(0, self.population_size - len(self.population)):
            self.create_agent(Status.Susceptible, social_stratum=i % 5)

        # Share the common wealth of 10^4 among the population, according each agent social stratum

        self.government.wealth = self.total_wealth/10

        for quintile in [0, 1, 2, 3, 4]:

            houses = [x for x in filter(lambda x: x.social_stratum == quintile, self.houses)]
            nhouses = len(houses)

            if nhouses == 0:
                self.create_house(social_stratum=quintile)
                houses = [self.houses[-1]]
                nhouses = 1

            total = lorenz_curve[quintile] * (5 * (self.total_wealth / 10))

            qty = max(1.0, np.sum([1.0 for a in self.business if a.social_stratum == quintile]))
            ag_share = total / qty
            for bus in filter(lambda x: x.social_stratum == quintile, self.business):
                bus.wealth = ag_share

            total = lorenz_curve[quintile] * (4 * (self.total_wealth/10))

            qty = max(1.0, np.sum([1 for a in self.population if
                                   a.social_stratum == quintile and a.economical_status == EconomicalStatus.Active]))
            ag_share = total / qty

            for agent in filter(lambda x: x.social_stratum == quintile, self.population):

                # distribute wealth

                if agent.economical_status == EconomicalStatus.Active:
                    agent.wealth = ag_share
                    agent.income = basic_income[agent.social_stratum] * self.minimum_income

                    # distribute employ
                    test = True
                    while test:
                        ix = np.random.randint(0, self.total_business)
                        if self.business[ix].social_stratum in [quintile, quintile+1]:
                            self.business[ix].hire(agent)
                            test = False
                        else:
                            test = True

                agent.expenses = self.minimum_expense

                #distribute habitation

                ix = np.random.randint(0, nhouses)
                house = houses[ix]
                house.append_mate(agent)


    def execute(self):
        self.iteration += 1

        bed = bed_time(self.iteration)
        work = work_time(self.iteration)
        free = free_time(self.iteration)
        lunch = lunch_time(self.iteration)
        new_dy = new_day(self.iteration)
        work_dy = work_day(self.iteration)
        new_mth = new_month(self.iteration)

        if new_dy:
            print("Day {}".format(self.iteration // 24))

        for agent in filter(lambda x: x.status != Status.Death, self.population):
            if bed:
                agent.move_to_home()
            elif lunch or free or not work_dy:
                agent.move_freely(self.amplitudes[agent.status])

            elif work_dy and work:
                agent.move_to_work()

            for bus in filter(lambda x: x != agent.employer, self.business):
                if distance(agent, bus) <= self.amplitudes[agent.status]:
                    bus.supply(agent)

            if new_dy:
                self.update(agent)


        if self.iteration > 1 and new_mth:
            for bus in self.business:
                bus.accounting(self.government)
            for house in self.houses:
                house.accounting(self.government)

        contacts = []

        for i in np.arange(0, self.population_size):
            for j in np.arange(i + 1, self.population_size):
                ai = self.population[i]
                aj = self.population[j]
                if ai.status == Status.Death or ai.status == Status.Death:
                    continue

                if distance(ai, aj) <= self.contagion_distance:
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
            agent.infected_time += 1

            indice = agent.age // 10 - 1 if agent.age > 10 else 0

            teste_sub = np.random.random()

            if agent.infected_status == InfectionSeverity.Asymptomatic:
                if age_hospitalization_probs[indice] > teste_sub:
                    agent.infected_status = InfectionSeverity.Hospitalization
                    agent.move_to(self.healthcare)
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
                agent.move_to_home()
                return

            if agent.infected_time > 20:
                agent.infected_time = 0
                agent.status = Status.Recovered_Immune
                agent.infected_status = InfectionSeverity.Asymptomatic

    def get_statistics(self, kind='ecom2'):
        if kind == 'ecom2':
            if self.statistics is None:
                self.statistics = {}
            if 'BusinessWealth' not in self.statistics:
                self.statistics['BusinessWealth'] = sum([b.wealth for b in self.business])
                self.statistics['BusinessStocks'] = sum([b.stocks for b in self.business])
                self.statistics['BusinessSales'] = sum([b.sales for b in self.business])
                self.statistics['HousesWealth'] = sum([b.wealth for b in self.houses])
                self.statistics['HousesExpenses'] = sum([b.expenses for b in self.houses])
            return self.statistics

        return super(GraphSimulation, self).get_statistics(kind)


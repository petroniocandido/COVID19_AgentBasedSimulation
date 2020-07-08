"""
Graph induced
"""

import numpy as np
from covid_abs.abs import *
from covid_abs.agents import *
from covid_abs.network.agents import EconomicalStatus, Business, House, Person
from covid_abs.network.util import new_day, work_day, new_month, bed_time, work_time, lunch_time, free_time


class GraphSimulation(Simulation):
    def __init__(self, **kwargs):
        super(GraphSimulation, self).__init__(**kwargs)
        self.total_population = kwargs.get('total_population', 0)
        self.total_business = kwargs.get('total_business', 10)
        self.business_distance = kwargs.get('business_distance', 10)
        self.government = None
        self.business = []
        self.houses = []
        self.healthcare = None
        self.homeless_rate = kwargs.get("homeless_rate", 0.01)
        self.unemployment_rate = kwargs.get("unemployment_rate", 0.09)
        self.homemates_avg = kwargs.get("homemates_avg", 3)
        self.homemates_std = kwargs.get("homemates_std", 1)
        self.iteration = -1
        self.callbacks = kwargs.get('callbacks', {})
        self.public_gdp_share = kwargs.get('public_gdp_share', 0.1)
        self.business_gdp_share = kwargs.get('business_gdp_share', 0.5)
        self.incubation_time = kwargs.get('incubation_time', 5)
        self.contagion_time = kwargs.get('contagion_time', 10)
        self.recovering_time = kwargs.get('recovering_time', 20)

    def register_callback(self, event, action):
        self.callbacks[event] = action

    def callback(self, event, *args):
        if event in self.callbacks:
            return self.callbacks[event](*args)

        return False

    def get_unemployed(self):
        return [p for p in self.population if p.is_unemployed()
                and p.status != Status.Death and p.infected_status == InfectionSeverity.Asymptomatic]

    def get_homeless(self):
        return [p for p in self.population if p.is_homeless()
                and p.status != Status.Death and p.infected_status == InfectionSeverity.Asymptomatic]

    def create_business(self, social_stratum=None):
        x, y = self.random_position()
        if social_stratum is None:
            social_stratum = int(np.random.rand(1) * 100 // 20)
        self.business.append(Business(x=x, y=y, status=Status.Susceptible, social_stratum=social_stratum,
                                      #fixed_expenses=(social_stratum+1)*self.minimum_expense
                                      #fixed_expenses=self.minimum_expense / (5 - social_stratum)
                                      environment=self
                                      ))

    def create_house(self, social_stratum=None):
        x, y = self.random_position()
        if social_stratum is None:
            social_stratum = int(np.random.rand(1) * 100 // 20)
        house = House(x=x, y=y, status=Status.Susceptible, social_stratum=social_stratum,
                                 #fixed_expenses=(social_stratum+1)*self.minimum_expense/(self.homemates_avg*10
                      environment=self)
        self.callback('on_create_house', house)
        self.houses.append(house)

    def create_agent(self, status, social_stratum=None, infected_time=0):
        """
        Create a new agent with the given status

        :param infected_time:
        :param social_stratum:
        :param status: a value of agents.Status enum
        :return: the newly created agent
        """

        age = int(np.random.beta(2, 5, 1) * 100)
        if social_stratum is None:
            social_stratum = int(np.random.rand(1) * 100 // 20)
        person = Person(age=age, status=status, social_stratum=social_stratum, infected_time=infected_time,
                        environment=self)
        self.callback('on_create_person', person)
        self.population.append(person)

    def initialize(self):
        """
        Initializate the Simulation by creating its population of agents
        """

        self.callback('on_initialize', self)

        x, y = self.random_position()
        self.healthcare = Business(x=x, y=y, status=Status.Susceptible, type=AgentType.Healthcare, environment=self)
        self.healthcare.fixed_expenses += self.minimum_expense * 3
        x, y = self.random_position()
        self.government = Business(x=x, y=y, status=Status.Susceptible, type=AgentType.Government,
                                   social_stratum=4, price=1.0, environment=self)
        self.government.fixed_expenses += self.population_size * (self.minimum_expense*0.05)

        #number of houses
        for i in np.arange(0, int(self.population_size // self.homemates_avg)):
            self.create_house(social_stratum=i % 5)

        # number of business
        for i in np.arange(0, self.total_business):
            self.create_business(social_stratum=5 - (i % 5))

        # Initial infected population
        for i in np.arange(0, int(self.population_size * self.initial_infected_perc)):
            self.create_agent(Status.Infected, infected_time=5)

        # Initial immune population
        for i in np.arange(0, int(self.population_size * self.initial_immune_perc)):
            self.create_agent(Status.Recovered_Immune)

        # Initial susceptible population
        for i in np.arange(0, self.population_size - len(self.population)):
            self.create_agent(Status.Susceptible, social_stratum=i % 5)

        # Share the common wealth of 10^4 among the population, according each agent social stratum

        self.government.wealth = self.total_wealth * self.public_gdp_share

        for quintile in [0, 1, 2, 3, 4]:

            _houses = [x for x in filter(lambda x: x.social_stratum == quintile, self.houses)]
            nhouses = len(_houses)

            if nhouses == 0:
                self.create_house(social_stratum=quintile)
                _houses = [self.houses[-1]]
                nhouses = 1

            if self.total_business > 5:
                btotal = lorenz_curve[quintile] * (self.total_wealth * self.business_gdp_share)
                bqty = max(1.0, np.sum([1.0 for a in self.business if a.social_stratum == quintile]))
            else:
                btotal = self.total_wealth * self.business_gdp_share
                bqty = self.total_business

            ag_share = btotal / bqty
            for bus in filter(lambda x: x.social_stratum == quintile, self.business):
                bus.wealth = ag_share

            ptotal = lorenz_curve[quintile] * self.total_wealth * (1 - (self.public_gdp_share + self.business_gdp_share))

            pqty = max(1.0, np.sum([1 for a in self.population if
                                   a.social_stratum == quintile and a.economical_status == EconomicalStatus.Active]))
            ag_share = ptotal / pqty

            for agent in filter(lambda x: x.social_stratum == quintile, self.population):

                # distribute wealth

                if agent.economical_status == EconomicalStatus.Active:
                    agent.wealth = ag_share
                    agent.incomes = basic_income[agent.social_stratum] * self.minimum_income

                    # distribute employ

                    unemployed_test = np.random.rand()

                    if unemployed_test >= self.unemployment_rate:
                        ix = np.random.randint(0, self.total_business)
                        self.business[ix].hire(agent)

                agent.expenses = basic_income[agent.social_stratum] * self.minimum_expense

                #distribute habitation

                homeless_test = np.random.rand()

                if not (quintile == 0 and homeless_test <= self.homeless_rate):
                    for kp in range(0, 5):
                        ix = np.random.randint(0, nhouses)
                        house = _houses[ix]
                        if house.size < self.homemates_avg + self.homemates_std:
                            house.append_mate(agent)
                            continue
                    if agent.house is None:
                        ix = np.random.randint(0, nhouses)
                        self.houses[ix].append_mate(agent)

        self.callback('post_initialize', self)

    def execute(self):

        self.iteration += 1

        if self.callback('on_execute', self):
            return

        #print(self.iteration)

        bed = bed_time(self.iteration)
        work = work_time(self.iteration)
        free = free_time(self.iteration)
        lunch = lunch_time(self.iteration)
        new_dy = new_day(self.iteration)
        work_dy = work_day(self.iteration)
        new_mth = new_month(self.iteration)

        #if new_dy:
        #    print("Day {}".format(self.iteration // 24))

        for agent in filter(lambda x: x.status != Status.Death, self.population):

            if not self.callback('on_person_move', agent):
                if bed:
                    agent.move_to_home()

                elif lunch or free or not work_dy:
                    agent.move_freely()

                elif work_dy and work:
                    agent.move_to_work()

            self.callback('post_person_move', agent)

            #agent.x = self._xclip(agent.x)
            #agent.y = self._yclip(agent.y)

            if new_dy:
                agent.update()

            if agent.infected_status == InfectionSeverity.Asymptomatic:
                for bus in filter(lambda x: x != agent.employer, self.business):
                    if distance(agent, bus) <= self.business_distance:
                        bus.supply(agent)

        for bus in filter(lambda b: b.open, self.business):
            if new_dy:
                bus.update()

            if self.iteration > 1 and new_mth:
                bus.accounting()

        for house in filter(lambda h: h.size > 0, self.houses):
            if new_dy:
                house.update()

            if self.iteration > 1 and new_mth:
                house.accounting()

        if new_dy:
            self.government.update()
            self.healthcare.update()

        if self.iteration > 1 and new_mth:
            self.government.accounting()

        contacts = []

        for i in np.arange(0, self.population_size):
            for j in np.arange(i + 1, self.population_size):
                ai = self.population[i]
                aj = self.population[j]
                if ai.status == Status.Death or ai.status == Status.Death:
                    continue

                if distance(ai, aj) <= self.contagion_distance:
                    contacts.append((i, j))

        for pair in contacts:
            ai = self.population[pair[0]]
            aj = self.population[pair[1]]
            self.contact(ai, aj)
            self.contact(aj, ai)

        self.statistics = None

        self.callback('post_execute', self)

    def contact(self, agent1, agent2):
        """
        Performs the actions needed when two agents get in touch.

        :param agent1: an instance of agents.Agent
        :param agent2: an instance of agents.Agent
        """

        if self.callback('on_contact', agent1, agent2):
            return

        if agent1.status == Status.Susceptible and agent2.status == Status.Infected:
            low = np.random.randint(-1, 1)
            up = np.random.randint(-1, 1)
            if agent2.infected_time >= self.incubation_time + low \
                    and agent2.infected_time <= self.contagion_time + up:
                contagion_test = np.random.random()
                #agent1.infection_status = InfectionSeverity.Exposed
                if contagion_test <= self.contagion_rate:
                    agent1.status = Status.Infected
                    agent1.infection_status = InfectionSeverity.Asymptomatic

        self.callback('post_contact', agent1, agent2)

    def get_statistics(self, kind='all'):
        if self.statistics is None:
            self.statistics = {}
            for quintile in [0, 1, 2, 3, 4]:
                #self.statistics['Q{}'.format(quintile + 1)] = np.sum(
                #    [a.wealth for a in self.population if a.social_stratum == quintile \
                #     and a.economical_status == EconomicalStatus.Active]) / self.total_wealth
                self.statistics['Q{}'.format(quintile + 1)] = np.sum(
                    h.wealth for h in self.houses if h.social_stratum == quintile
                ) / self.total_wealth
            self.statistics['Business'] = np.sum([b.wealth for b in self.business]) / self.total_wealth
            self.statistics['Government'] = self.government.wealth / self.total_wealth

            for status in Status:
                self.statistics[status.name] = np.sum(
                    [1 for a in self.population if a.status == status]) / self.population_size

            for infected_status in filter(lambda x: x != InfectionSeverity.Exposed, InfectionSeverity):
                self.statistics[infected_status.name] = np.sum([1 for a in self.population if
                                                                a.infected_status == infected_status and
                                                                a.status != Status.Death]) / self.population_size

        return self.filter_stats(kind)


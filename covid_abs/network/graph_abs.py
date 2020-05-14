"""
Graph induced
"""

from covid_abs.abs import *
from covid_abs.network.agents import EconomicalStatus, Business, House, Person
from covid_abs.network.util import new_day, work_day, new_month, bed_time, work_time, lunch_time, free_time


class GraphSimulation(Simulation):
    def __init__(self, **kwargs):
        super(GraphSimulation, self).__init__(**kwargs)
        self.total_population = kwargs.get('total_population', 0)
        self.total_business  = kwargs.get('total_business', 10)
        self.government = None
        self.business = []
        self.houses = []
        self.healthcare = None
        self.homeless_rate = kwargs.get("homeless_rate", 0.01)
        self.unemployment_rate = kwargs.get("unemployment_rate", 0.1)
        self.homemates_avg = kwargs.get("homemates_avg", 3)
        self.homemates_std = kwargs.get("homemates_std", 1)
        self.iteration = -1
        self.triggers_business = kwargs.get("triggers_business", [])
        "A dictionary with conditional changes in the Business attributes"
        self.triggers_government = kwargs.get("triggers_government", [])
        "A dictionary with conditional changes in the Government attributes"

    def append_trigger_business(self, condition, attribute, action):
        self.triggers_business.append({'condition': condition, 'attribute': attribute, 'action': action})

    def append_triggers_government(self, condition, attribute, action):
        self.triggers_government.append({'condition': condition, 'attribute': attribute, 'action': action})

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

                    unemployed_test = np.random.rand()

                    if unemployed_test > self.unemployment_rate:
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

                homeless_test = np.random.rand()

                if homeless_test > self.homeless_rate:
                    test = True
                    while test:
                        ix = np.random.randint(0, nhouses)
                        house = houses[ix]
                        if house.social_stratum == agent.social_stratum and\
                                house.size < self.homemates_avg + self.homemates_std:
                            house.append_mate(agent)
                            test = False

    def execute(self):
        self.iteration += 1

        move_home_triggers = [k for k in self.triggers_population if k['attribute'] == 'move_home']
        move_work_triggers = [k for k in self.triggers_population if k['attribute'] == 'move_work']
        move_freely_triggers = [k for k in self.triggers_population if k['attribute'] == 'move_freely']
        other_triggers = [k for k in self.triggers_population if not k['attribute'].startswith('move')]

        bed = bed_time(self.iteration)
        work = work_time(self.iteration)
        free = free_time(self.iteration)
        lunch = lunch_time(self.iteration)
        new_dy = new_day(self.iteration)
        work_dy = work_day(self.iteration)
        new_mth = new_month(self.iteration)

        if new_dy:
            print("Day {}".format(self.iteration // 24))

        if len(self.triggers_government) > 0:
            for trigger in self.triggers_government:
                if trigger['condition'](self.government):
                    attr = trigger['attribute']
                    self.government.__dict__[attr] = trigger['action'](self.government)

        if len(self.triggers_business) > 0:
            for trigger in self.triggers_simulation:
                for bus in self.business:
                    if trigger['condition'](bus):
                        attr = trigger['attribute']
                        bus.__dict__[attr] = trigger['action'](bus)

        for agent in filter(lambda x: x.status != Status.Death, self.population):
            if bed:
                agent.move_to_home(triggers=move_home_triggers)

            elif lunch or free or not work_dy:
                agent.move_freely(self.amplitudes[agent.status], triggers=move_freely_triggers)

            elif work_dy and work:
                agent.move_to_work(triggers=move_work_triggers)

            agent.x = self._xclip(agent.x)
            agent.y = self._yclip(agent.y)

            for trigger in other_triggers:
                if trigger['condition'](agent):
                    attr = trigger['attribute']
                    agent.__dict__[attr] = trigger['action'](agent.__dict__[attr])

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

        if len(self.triggers_simulation) > 0:
            for trigger in self.triggers_simulation:
                if trigger['condition'](self):
                    attr = trigger['attribute']
                    self.__dict__[attr] = trigger['action'](self.__dict__[attr])

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

        elif kind == 'ecom3':
            if self.statistics is None:
                self.statistics = {
                    'AvgHousemates': np.average([h.size for h in self.houses]),
                    'AvgEmployees': np.average([h.num_employees for h in self.business]),
                    'NumUnemployed': np.sum([1 for h in self.population if h.employer is None
                                             and h.economical_status == EconomicalStatus.Active]),
                    'NumHomeless': np.sum([1 for h in self.population if h.house is None]),
                    'NumInactive': np.sum([1 for h in self.population if h.economical_status == EconomicalStatus.Inactive])
                }
        else:
            return super(GraphSimulation, self).get_statistics(kind)

        return self.statistics


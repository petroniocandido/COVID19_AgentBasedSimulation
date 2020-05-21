"""
Graph induced
"""

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
        self.population_move_triggers = kwargs.get('population_move_triggers', [])
        self.population_move_home_triggers = kwargs.get('population_move_home_triggers', [])
        self.population_move_work_triggers = kwargs.get('population_move_work_triggers', [])
        self.population_move_freely_triggers = kwargs.get('population_move_freely_triggers', [])
        self.population_other_triggers = kwargs.get('population_other_triggers', [])

    def apply_business(self, filter_attribute, filter_value, target_attribute, target_value):
        for bus in self.business:
            if bus.__dict__[filter_attribute] == filter_value:
                bus.__dict__[target_attribute] == target_value

    def apply_government(self, filter_attribute, filter_value, target_attribute, target_value):
        if self.government.__dict__[filter_attribute] == filter_value:
            self.government.__dict__[target_attribute] == target_value

    def append_trigger_population(self, condition, attribute, action):
        """
        Append a conditional change in the population attributes

        :param condition: a lambda function that receives the current agent instance and returns a boolean
        :param attribute: string, the attribute name of the agent which will be changed
        :param action: a lambda function that receives the current agent instance and returns the new
        value of the attribute
        """
        #self.triggers_population.append({'condition': condition, 'attribute': attribute, 'action': action})

        if attribute == 'move':
            self.population_move_triggers.append({'condition': condition, 'attribute': attribute, 'action': action})
        elif attribute == 'move_home':
            self.population_move_home_triggers.append({'condition': condition, 'attribute': attribute, 'action': action})
        elif attribute == 'move_work':
            self.population_move_work_triggers.append({'condition': condition, 'attribute': attribute, 'action': action})
        elif attribute == 'move_freely':
            self.population_move_freely_triggers.append({'condition': condition, 'attribute': attribute, 'action': action})
        else:
            self.population_other_triggers.append({'condition': condition, 'attribute': attribute, 'action': action})

    def remove_trigger_population(self, attribute, index=-1):
        if attribute == 'move':
            if len(self.population_move_triggers) > 0:
                self.population_move_triggers.pop(index)
        elif attribute == 'move_home':
            if len(self.population_move_home_triggers) > 0:
                self.population_move_home_triggers.pop(index)
        elif attribute == 'move_work':
            if len(self.population_move_work_triggers) > 0:
                self.population_move_work_triggers.pop(index)
        elif attribute == 'move_freely':
            if len(self.population_move_freely_triggers) > 0:
                self.population_move_freely_triggers.pop(index)
        else:
            if len(self.population_other_triggers) > 0:
                self.population_other_triggers.pop(index)

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
                                      ))

    def create_house(self, social_stratum=None):
        x, y = self.random_position()
        if social_stratum is None:
            social_stratum = int(np.random.rand(1) * 100 // 20)
        self.houses.append(House(x=x, y=y, status=Status.Susceptible, social_stratum=social_stratum,
                                 #fixed_expenses=(social_stratum+1)*self.minimum_expense/(self.homemates_avg*10
                                 ))

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
        self.population.append(Person(age=age, status=status, social_stratum=social_stratum, infected_time=infected_time))

    def initialize(self):
        """
        Initializate the Simulation by creating its population of agents
        """
        x, y = self.random_position()
        self.healthcare = Business(x=x, y=y, status=Status.Susceptible, type=AgentType.Healthcare)
        self.healthcare.fixed_expenses += self.minimum_expense * 3
        x, y = self.random_position()
        self.government = Business(x=x, y=y, status=Status.Susceptible, type=AgentType.Government,
                                   social_stratum=4, price=1.0)
        self.government.fixed_expenses += self.population_size * (self.minimum_expense*0.05)

        #number of houses
        for i in np.arange(0, int(self.population_size // self.homemates_avg)):
            self.create_house(social_stratum=i % 5)

        # number of business
        for i in np.arange(0, self.total_business):
            self.create_business(social_stratum=i % 5)

        # Initial infected population
        for i in np.arange(0, int(self.population_size * self.initial_infected_perc)):
            self.create_agent(Status.Infected, infected_time=1)

        # Initial immune population
        for i in np.arange(0, int(self.population_size * self.initial_immune_perc)):
            self.create_agent(Status.Recovered_Immune)

        # Initial susceptible population
        for i in np.arange(0, self.population_size - len(self.population)):
            self.create_agent(Status.Susceptible, social_stratum=i % 5)

        # Share the common wealth of 10^4 among the population, according each agent social stratum

        d = self.total_wealth/10

        self.government.wealth = d

        for quintile in [0, 1, 2, 3, 4]:

            _houses = [x for x in filter(lambda x: x.social_stratum == quintile, self.houses)]
            nhouses = len(_houses)

            if nhouses == 0:
                self.create_house(social_stratum=quintile)
                _houses = [self.houses[-1]]
                nhouses = 1

            btotal = lorenz_curve[quintile] * (5 * d)

            bqty = max(1.0, np.sum([1.0 for a in self.business if a.social_stratum == quintile]))
            ag_share = btotal / bqty
            for bus in filter(lambda x: x.social_stratum == quintile, self.business):
                bus.wealth = ag_share

            ptotal = lorenz_curve[quintile] * (4 * d)

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
                        for kp in range(5):
                            ix = np.random.randint(0, self.total_business)
                            if self.business[ix].social_stratum in [quintile, quintile+1]:
                                self.business[ix].hire(agent)
                                continue
                        if agent.employer is None:
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

    def pull_agent_trigger(self, agent, triggers):
        for trigger in triggers:
            if trigger['condition'](agent, self):
                if trigger['attribute'].startswith('move'):
                    agent.x, agent.y = trigger['action'](agent, self)
                else:
                    attr = trigger['attribute']
                    agent.__dict__[attr] = trigger['action'](agent.__dict__[attr])
                return True
        return False

    def execute(self):
        self.iteration += 1

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
            amplitude = self.amplitudes[agent.status]

            if not self.pull_agent_trigger(agent, self.population_move_triggers):
                if bed and not self.pull_agent_trigger(agent, self.population_move_home_triggers):
                    agent.move_to_home(amplitude)

                elif (lunch or free or not work_dy) and \
                        not self.pull_agent_trigger(agent, self.population_move_freely_triggers):
                    agent.move_freely(amplitude)

                elif (work_dy and work) and not self.pull_agent_trigger(agent, self.population_move_work_triggers):
                    agent.move_to_work(amplitude)

            #agent.x = self._xclip(agent.x)
            #agent.y = self._yclip(agent.y)

            self.pull_agent_trigger(agent, self.population_other_triggers)

            for bus in filter(lambda x: x != agent.employer, self.business):
                if distance(agent, bus) <= self.business_distance:
                    bus.supply(agent)

            if new_dy:
                agent.update(self)

        for bus in self.business:
            if new_dy:
                bus.update(self)

            if self.iteration > 1 and new_mth:
                bus.accounting(self)

        for house in self.houses:
            if new_dy:
                house.update(self)

            if self.iteration > 1 and new_mth:
                house.accounting(self)

        if new_dy:
            self.government.update(self)
            self.healthcare.update(self)

        if self.iteration > 1 and new_mth:
            self.government.accounting(self)

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

        if len(self.triggers_simulation) > 0:
            for trigger in self.triggers_simulation:
                if trigger['condition'](self):
                    attr = trigger['attribute']
                    if attr == 'execute':
                        trigger['action'](self)
                    else:
                        self.__dict__[attr] = trigger['action'](self)

        self.statistics = None

    def contact(self, agent1, agent2):
        """
        Performs the actions needed when two agents get in touch.

        :param agent1: an instance of agents.Agent
        :param agent2: an instance of agents.Agent
        """

        if agent1.status == Status.Susceptible and agent2.status == Status.Infected:
            if 1 < agent2.infected_time < 15:
                contagion_test = np.random.random()
                agent1.infection_status = InfectionSeverity.Exposed
                if contagion_test <= self.contagion_rate:
                    agent1.status = Status.Infected
                    agent1.infection_status = InfectionSeverity.Asymptomatic

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


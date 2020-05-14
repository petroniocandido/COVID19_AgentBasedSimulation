from enum import Enum

import numpy as np

from covid_abs.agents import Agent, AgentType, InfectionSeverity


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

        if self.age > 16 and self.age <= 65:
            self.economical_status = EconomicalStatus.Active

    def demand(self, value = 0.0):
        """Expense for product/services"""
        if self.house is not None:
            self.house.demand(value)
        else:
            self.wealth -= value

    def supply(self, value = 0.0):
        """Income for work"""
        if self.house is not None:
            self.house.supply(value)
        else:
            self.wealth += value

    def move_to_work(self, triggers=[]):
        if self.infected_status != InfectionSeverity.Asymptomatic:
            return

        for trigger in triggers:
            if trigger['condition'](self):
                self.x, self.y = trigger['action'](self)
                return

        if self.economical_status == EconomicalStatus.Active:
            if self.employer is not None and self.employer.open:
                self.x = int(self.employer.x + np.random.normal(0.0, 0.25, 1))
                self.y = int(self.employer.y + np.random.normal(0.0, 0.25, 1))
                self.employer.checkin(self)
            elif self.employer is None:
                self.x = int(self.x + np.random.normal(0.0, 1, 1))
                self.y = int(self.y + np.random.normal(0.0, 1, 1))

    def move_to_home(self, triggers=[]):
        if self.infected_status != InfectionSeverity.Asymptomatic:
            return

        for trigger in triggers:
            if trigger['condition'](self):
                self.x, self.y = trigger['action'](self)
                return

        if self.house is not None:
            self.house.checkin(self)
            self.x = int(self.house.x + np.random.normal(0.0, 0.25, 1))
            self.y = int(self.house.y + np.random.normal(0.0, 0.25, 1))
        else:
            self.wealth -= self.incomes / 720
            self.x = int(self.x + np.random.normal(0.0, 1, 1))
            self.y = int(self.y + np.random.normal(0.0, 1, 1))

    def move_freely(self, amplitude, triggers=[]):
        if self.infected_status != InfectionSeverity.Asymptomatic:
            return

        for trigger in triggers:
            if trigger['condition'](self):
                self.x, self.y = trigger['action'](self)
                return

        x,y = np.random.normal(0, amplitude, 2)
        self.x = int(self.x + x)
        self.y = int(self.y + y)

    def move_to(self, agent, triggers=[]):

        for trigger in triggers:
            if trigger['condition'](self):
                self.x, self.y = trigger['action'](agent)
                return

        self.x = int(agent.x + np.random.normal(0.0, 0.25, 1))
        self.y = int(agent.y + np.random.normal(0.0, 0.25, 1))
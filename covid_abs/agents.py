from enum import Enum


class Status(Enum):
    """
    Agent status, following the SIR model
    """
    Susceptible = 's'
    Infected = 'i'
    Recovered_Immune = 'c'
    Death = 'm'


class InfectionSeverity(Enum):
    Asymptomatic = 'a'
    Hospitalization = 'h'
    Severe = 'g'


class Agent(object):
  def __init__(self, **kwargs):
    self.x = kwargs.get('x',0)
    self.y = kwargs.get('y',0)
    self.status = kwargs.get('status',Status.Susceptible)
    self.infected_status = InfectionSeverity.Asymptomatic
    self.infected_time = 0
    self.age = kwargs.get('age',0)
    self.social_stratum = kwargs.get('social_stratum',0)
    self.wealth = kwargs.get('wealth',0.0)

  def get_description(self):
    if self.status == Status.Infected:
      return "{}({})".format(self.status.name, self.infected_status.name)
    else:
      return self.status.name

  def __str__(self):
    return str(self.status.name)
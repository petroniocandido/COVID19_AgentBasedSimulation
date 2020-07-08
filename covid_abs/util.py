import numpy as np

def change_attribute(object, attribute, new_value):
    object.__dict__[attribute] = new_value
    return False

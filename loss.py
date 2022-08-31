from utils import *

def MSE(predictions, actual):
    return ((predictions - actual) ** 2 / 2).mean()

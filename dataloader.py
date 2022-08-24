import numpy as np
import pandas as pd
import os

class DataLoader:
    def __init__(self) -> None:
        pass
    
    def __iter__(self, batchSize):
        pass

    def __next__(self):
        pass

    def __call__(self, batchSize):
        return self.__iter__(batchSize)


class MathFunc(DataLoader):
    def __init__(self, func, xmin, xmax, size, batchSize) -> None:
        self.func = func
        self.xmin = xmin
        self.xmax = xmax
        self.size = size
        self.batchSize = batchSize
    
    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        b = min(self.batchSize, self.size - self.i)
        self.i += b
        if b <= 0:
            raise StopIteration
        input = np.random.uniform(self.xmin, self.xmax, (b, 1))
        output = self.func(input)
        return input, output

    def __call__(self):
        return self.__iter__()
